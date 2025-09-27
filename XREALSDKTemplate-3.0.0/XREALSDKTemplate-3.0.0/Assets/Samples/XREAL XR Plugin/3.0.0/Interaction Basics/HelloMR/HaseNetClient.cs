using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

public class HaseNetClient : MonoBehaviour
{
    public enum Mode { Live, ReplayFolder }

    [Header("Server")]
    public string serverWs = "ws://192.168.1.100:8765"; // <-- set to your laptop IP
    public Mode mode = Mode.Live;

    [Header("Live mode input")]
    public WindowMux mux;                 // assign if Mode = Live

    [Header("Replay mode input (device folder with *_imu.bin, *_mfcc.bin, *_meta.json)")]
    public string replayFolder = "";      // leave empty to use Application.persistentDataPath + "/hase_windows"
    public float replayDelaySec = 0.1f;   // pause between sends

    [Header("UI (optional)")]
    public PromptSlide prompt;            // to display the server prompt text

    ClientWebSocket _ws;
    CancellationTokenSource _cts;
    readonly ConcurrentQueue<Action> _mainThread = new ConcurrentQueue<Action>();
    bool _subscribed;
    [Header("Network receive")]
    public bool receiveServerPrompts = false; // default off

    void Start() => _ = ConnectAndRun();

    async Task ConnectAndRun()
    {
        try
        {
            _ws = new ClientWebSocket();
            _cts = new CancellationTokenSource();
            await _ws.ConnectAsync(new Uri(serverWs), _cts.Token);
            Debug.Log("[HASE-NET] Connected " + serverWs);
            var hello = new ArraySegment<byte>(Encoding.UTF8.GetBytes(
            "{\"client\":\"beampro\",\"msg\":\"hello\"}"));
            await _ws.SendAsync(hello, WebSocketMessageType.Text, true, _cts.Token);


            // start receive loop
            _ = Task.Run(ReceiveLoop);

            if (mode == Mode.Live && !_subscribed && mux)
            {
                Debug.Log("[HASE-NET] Subscribing to mux.OnHaseWindow");
                mux.OnHaseWindow += OnWindow;
                _subscribed = true;
            }
            else
            {
                Debug.LogWarning($"[HASE-NET] Not subscribing. mode={mode}, mux={(mux ? "set" : "NULL")}, _subscribed={_subscribed}");
            }




        }
        catch (Exception e)
        {
            Debug.LogError("[HASE-NET] Connect failed: " + e);
        }
    }

    void OnDestroy()
    {
        if (_subscribed && mux) mux.OnHaseWindow -= OnWindow;
        _subscribed = false;
        try
        {
            if (mode == Mode.Live && mux) mux.OnHaseWindow -= OnWindow;
            _cts?.Cancel();
            _ws?.Abort();
            _ws?.Dispose();
        }
        catch { }
    }
    void Awake()
    {
        var all = FindObjectsOfType<HaseNetClient>(true);
        Debug.Log($"[HASE-NET] clients in scene: {all.Length}, thisId={GetInstanceID()}, mux? {(mux ? "yes" : "no")} muxId={(mux ? mux.GetInstanceID().ToString() : "null")}");
    }


    // ----------- LIVE MODE -----------
    void OnWindow(float[] imu, float[] mfcc, double t0)
    {
        Debug.Log($"[HASE-NET] OnWindow imu={imu?.Length} mfcc={mfcc?.Length} t0={t0}");
        if (_ws == null || _ws.State != WebSocketState.Open)
        {
            Debug.LogWarning("[HASE-NET] WebSocket not open; skip send");
            return;
        }
        _ = SendHaseFrame(imu, mfcc, t0);
    }

    // ----------- REPLAY MODE -----------
    IEnumerator ReplayCoroutine()
    {
        string root = replayFolder;
        if (string.IsNullOrEmpty(root))
            root = Path.Combine(Application.persistentDataPath, "hase_windows");

        if (!Directory.Exists(root))
        {
            Debug.LogWarning("[HASE-NET] Replay folder not found: " + root);
            yield break;
        }

        // Prefer bin+json triplets; fall back to *_window.json or *_window.json.gz if present.
        var metas = new List<string>(Directory.GetFiles(root, "*_meta.json"));
        metas.Sort(StringComparer.Ordinal);

        if (metas.Count == 0)
        {
            // try combined JSON (gz) files
            var gz = new List<string>(Directory.GetFiles(root, "*_window.json.gz"));
            gz.Sort(StringComparer.Ordinal);
            foreach (var fn in gz)
            {
                if (_ws.State != WebSocketState.Open) yield break;
                if (TryLoadWindowJson(fn, out var t0, out var imu, out var mfcc))
                    _ = SendHaseFrame(imu, mfcc, t0);
                yield return new WaitForSecondsRealtime(replayDelaySec);
            }
            // try plain json
            var js = new List<string>(Directory.GetFiles(root, "*_window.json"));
            js.Sort(StringComparer.Ordinal);
            foreach (var fn in js)
            {
                if (_ws.State != WebSocketState.Open) yield break;
                if (TryLoadWindowJson(fn, out var t0, out var imu, out var mfcc))
                    _ = SendHaseFrame(imu, mfcc, t0);
                yield return new WaitForSecondsRealtime(replayDelaySec);
            }
            yield break;
        }

        foreach (var meta in metas)
        {
            if (_ws.State != WebSocketState.Open) yield break;

            // epoch prefix (integer seconds)
            var epoch = Path.GetFileName(meta).Split('_')[0];

            // load meta
            double t0 = 0;
            try
            {
                var txt = File.ReadAllText(meta);
                var key = "\"epoch_start_s\":";
                var i = txt.IndexOf(key, StringComparison.Ordinal);
                if (i >= 0)
                {
                    var j = txt.IndexOfAny(new[] { ',', '}', '\n', '\r', ' ' }, i + key.Length);
                    var num = txt.Substring(i + key.Length, (j > i ? j : txt.Length) - (i + key.Length));
                    t0 = double.Parse(num, System.Globalization.CultureInfo.InvariantCulture);
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning("[HASE-NET] Failed to read meta: " + e.Message);
            }

            var imuPath = Path.Combine(root, $"{epoch}_imu.bin");
            var mfccPath = Path.Combine(root, $"{epoch}_mfcc.bin");
            if (!File.Exists(imuPath) || !File.Exists(mfccPath))
            {
                Debug.LogWarning($"[HASE-NET] Missing bin(s) for epoch {epoch}");
                continue;
            }

            // read floats (little-endian float32)
            var imu = ReadFloats(imuPath);
            var mfcc = ReadFloats(mfccPath);

            if (imu.Length != 800 * 6 || mfcc.Length != 430 * 13)
                Debug.LogWarning($"[HASE-NET] Unexpected shapes imu={imu.Length} mfcc={mfcc.Length}");

            _ = SendHaseFrame(imu, mfcc, t0);
            yield return new WaitForSecondsRealtime(replayDelaySec);
        }
    }
    void Update() { while (_mainThread.TryDequeue(out var a)) a?.Invoke(); }

    static float[] ReadFloats(string path)
    {
        var bytes = File.ReadAllBytes(path);
        int n = bytes.Length / 4;
        var arr = new float[n];
        Buffer.BlockCopy(bytes, 0, arr, 0, bytes.Length);
        return arr;
    }

    bool TryLoadWindowJson(string path, out double t0, out float[] imu, out float[] mfcc)
    {
        t0 = 0; imu = null; mfcc = null;
        try
        {
            string json;
            if (path.EndsWith(".gz", StringComparison.OrdinalIgnoreCase))
            {
                using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
                using var gz = new GZipStream(fs, CompressionMode.Decompress);
                using var ms = new MemoryStream();
                gz.CopyTo(ms);
                json = Encoding.UTF8.GetString(ms.ToArray());
            }
            else
            {
                json = File.ReadAllText(path);
            }

            // naive parsing (fast)
            t0 = ParseNumber(json, "\"epoch_start_s\"");
            imu = ParseFloatArray(json, "\"imu\"");
            mfcc = ParseFloatArray(json, "\"mfcc\"");
            return (imu != null && mfcc != null);
        }
        catch (Exception e)
        {
            Debug.LogWarning("[HASE-NET] JSON load failed: " + e.Message);
            return false;
        }
    }

    static double ParseNumber(string json, string key)
    {
        int i = json.IndexOf(key, StringComparison.Ordinal);
        if (i < 0) throw new KeyNotFoundException($"Key not found: {key}");

        int s = json.IndexOf(':', i);
        if (s < 0) throw new FormatException($"No ':' after key {key}");
        s++;

        // Skip spaces/tabs/newlines
        while (s < json.Length && char.IsWhiteSpace(json[s])) s++;

        int e = s;
        while (e < json.Length)
        {
            char c = json[e];
            if (!(char.IsDigit(c) || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E'))
                break;
            e++;
        }

        var token = json.Substring(s, e - s);
        if (!double.TryParse(token, NumberStyles.Float, CultureInfo.InvariantCulture, out var val))
            throw new FormatException($"Could not parse number for {key}: '{token}'");
        return val;
    }
    static float[] ParseFloatArray(string json, string key)
    {
        int i = json.IndexOf(key, StringComparison.Ordinal);
        if (i < 0) return null;
        int l = json.IndexOf('[', i);
        int r = json.IndexOf(']', l + 1);
        var body = json.Substring(l + 1, r - l - 1).Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
        var a = new float[body.Length];
        for (int k = 0; k < body.Length; k++)
            a[k] = float.Parse(body[k], System.Globalization.CultureInfo.InvariantCulture);
        return a;
    }

    // ----------- SEND / RECEIVE -----------
    async Task SendHaseFrame(float[] imu, float[] mfcc, double t0)
    {
        try
        {
            // HASE v1 header (little-endian)
            // uint32 'HASE', u8 ver=1, u8 flags=0, u16 res=0,
            // double t0,
            // u32 imu_hz=40, u32 imu_n=800, u32 imu_c=6, u32 imu_count,
            // u32 mfcc_n=430, u32 mfcc_c=13, u32 mfcc_count
            using var ms = new MemoryStream(64 + (imu.Length + mfcc.Length) * 4);
            using var bw = new BinaryWriter(ms);
            bw.Write(0x48415345u); // 'HASE'
            bw.Write((byte)1);
            bw.Write((byte)0);
            bw.Write((ushort)0);
            bw.Write(t0);

            bw.Write((uint)40);              // imu_hz
            bw.Write((uint)800);             // imu_n
            bw.Write((uint)6);               // imu_c
            bw.Write((uint)imu.Length);      // imu_count

            bw.Write((uint)430);             // mfcc_n
            bw.Write((uint)13);              // mfcc_c
            bw.Write((uint)mfcc.Length);     // mfcc_count

            foreach (var v in imu) bw.Write(v);
            foreach (var v in mfcc) bw.Write(v);
            bw.Flush();

            var seg = new ArraySegment<byte>(ms.ToArray());
            await _ws.SendAsync(seg, WebSocketMessageType.Binary, true, _cts.Token);
            // Debug.Log($"[HASE-NET] Sent window t0={t0:F0}");
        }
        catch (Exception e)
        {
            Debug.LogWarning("[HASE-NET] Send failed: " + e.Message);
        }
    }

    async Task ReceiveLoop()
    {
        var buf = new byte[8192];
        while (_ws != null && _ws.State == WebSocketState.Open)
        {
            WebSocketReceiveResult r;
            var ms = new MemoryStream();
            do
            {
                var seg = new ArraySegment<byte>(buf);
                r = await _ws.ReceiveAsync(seg, _cts.Token);
                if (r.MessageType == WebSocketMessageType.Close) { Debug.Log("[HASE-NET] Server closed"); return; }
                ms.Write(seg.Array, seg.Offset, r.Count);
            } while (!r.EndOfMessage);

            if (r.MessageType == WebSocketMessageType.Text)
            {
                var json = Encoding.UTF8.GetString(ms.ToArray());
                HandleServerJson(json);
            }
            // (if server ever sent binary back, you could handle it here)
        }
    }

    void HandleServerJson(string json)
    {
        Debug.Log($"[HaseNetClient] Received JSON from server: {json}");
        if (!receiveServerPrompts) return;     // <-- ignore server prompts entirely

        // 1. Extract the value of the "type" key.
        string messageType = ExtractString(json, "\"type\"");
        Debug.Log(messageType);
        // 2. Check if the value is "prompt".
        if (messageType != "prompt")
        {
            // If it's not a prompt message, ignore it.
            return;
        }
        string text = ExtractString(json, "\"text\"");
        bool hase = ExtractBool(json, "\"hase_window\"");
        double t0 = ExtractNumber(json, "\"t0\"");
        Debug.Log(hase);
        // If you ever enable this flag later, also use the main-thread queue:
        if (hase && prompt && !string.IsNullOrEmpty(text))
        {
            Debug.Log("[HaseNetClient] Conditions met. Enqueuing prompt to main thread.");
            _mainThread.Enqueue(() => prompt.ShowOnce(text));
        }
    }




    static string ExtractString(string json, string key)
    {
        int i = json.IndexOf(key, StringComparison.Ordinal);
        if (i < 0) return null;
        int q1 = json.IndexOf('"', i + key.Length);
        int q2 = json.IndexOf('"', q1 + 1);
        return json.Substring(q1 + 1, q2 - q1 - 1);
    }
    static bool ExtractBool(string json, string key)
    {
        int i = json.IndexOf(key, StringComparison.Ordinal);
        if (i < 0) return false;
        int s = json.IndexOf(':', i) + 1;
        int e = json.IndexOfAny(new[] { ',', '}', '\n', '\r', ' ' }, s);
        var tok = json.Substring(s, (e > s ? e : json.Length) - s).Trim();
        return tok.StartsWith("t") || tok.StartsWith("T");
    }
    static double ExtractNumber(string json, string key) => ParseNumber(json, key);
    static double ExtractNumberFromObject(string json, string objectKey, string targetKey)
    {
        // Find the start of the parent object, e.g., "scores":{
        int objectIndex = json.IndexOf(objectKey, StringComparison.Ordinal);
        if (objectIndex < 0) return 0;

        int objectStart = json.IndexOf('{', objectIndex);
        if (objectStart < 0) return 0;

        // Find the end of the parent object by matching curly braces
        int braceCount = 1;
        int objectEnd = objectStart + 1;
        while (objectEnd < json.Length && braceCount > 0)
        {
            if (json[objectEnd] == '{') braceCount++;
            if (json[objectEnd] == '}') braceCount--;
            objectEnd++;
        }

        if (braceCount != 0) return 0; // JSON is malformed

        // We now have the substring for the scores object, e.g., {"p_social":...}
        string objectJson = json.Substring(objectStart, objectEnd - objectStart);

        // Now we can safely search for the target key inside this smaller string
        return ParseNumber(objectJson, targetKey);
    }
}
