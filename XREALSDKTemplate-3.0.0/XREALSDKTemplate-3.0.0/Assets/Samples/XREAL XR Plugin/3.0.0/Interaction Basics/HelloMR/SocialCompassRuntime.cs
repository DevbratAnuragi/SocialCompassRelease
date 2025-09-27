using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class SocialCompassRuntime : MonoBehaviour
{
    [Header("Inputs")]
    public WindowMux mux;  // assign

    [Header("Artifacts in StreamingAssets")]
    public string thresholdSocialJson = "threshold_final.json";
    public string thresholdArousalJson = "arousal_threshold.json";
    public string arousalAnchorsJson = "arousal_text_anchors.json";

    [Header("Motion gate (optional)")]
    public bool useMotionGate = true;
    public float motionCutoff = 0.35f; // tune later (std L2 across 6 axes)

    [Header("Debounce events (optional)")]
    public bool emitEvents = true;
    public float mergeGapSecs = 60f;

    [Header("Output hook (UI / network / logs)")]
    public PromptSlide prompt; // optional: show prompt on hase_window

    float thetaSocial = 0.5f;
    float thetaArousal = 0.0f;
    float[] eHigh, eLow; // 512 each (L2-normed)

    // Debounce state
    class EventAcc { public double start, end; public int n; }
    EventAcc currentEvent;

    void OnEnable()
    {
        LoadArtifacts();
        if (mux) mux.OnHaseWindow += OnWindow;
    }
    void OnDisable()
    {
        if (mux) mux.OnHaseWindow -= OnWindow;
    }

    void LoadArtifacts()
    {
        try
        {
            string root = Application.streamingAssetsPath;
            thetaSocial = LoadThreshold(Path.Combine(root, thresholdSocialJson));
            thetaArousal = LoadThreshold(Path.Combine(root, thresholdArousalJson));
            (eHigh, eLow) = LoadAnchors(Path.Combine(root, arousalAnchorsJson));
            Debug.Log($"[SC] θ_social={thetaSocial:F3} θ_arousal={thetaArousal:F3} anchors={(eHigh?.Length ?? 0)}");
        }
        catch (Exception e)
        {
            Debug.LogWarning("[SC] Artifact load failed, using defaults. " + e.Message);
        }
    }

    float LoadThreshold(string path)
    {
        var txt = ReadText(path);
        int k = txt.IndexOf(":");
        int j = txt.IndexOf("}", k + 1);
        var num = txt.Substring(k + 1, j - k - 1).Trim().TrimEnd(',');
        return float.Parse(num, System.Globalization.CultureInfo.InvariantCulture);
    }

    (float[], float[]) LoadAnchors(string path)
    {
        var txt = ReadText(path);
        var hi = ParseArray(txt, "\"e_high\"");
        var lo = ParseArray(txt, "\"e_low\"");
        return (hi, lo);
    }

    static string ReadText(string path)
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        // StreamingAssets can be inside APK; use UnityWebRequest for robustness
        using (var www = UnityEngine.Networking.UnityWebRequest.Get(path))
        {
            var op = www.SendWebRequest();
            while (!op.isDone) {}
#if UNITY_2020_2_OR_NEWER
            if (www.result != UnityEngine.Networking.UnityWebRequest.Result.Success)
#else
            if (www.isNetworkError || www.isHttpError)
#endif
                throw new IOException(www.error);
            return www.downloadHandler.text;
        }
#else
        return File.ReadAllText(path);
#endif
    }

    static float[] ParseArray(string json, string key)
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

    void OnWindow(float[] imuFlat, float[] mfccFlat, double t0)
    {
        // 1) optional motion gate
        float motion = MotionEnergy(imuFlat);
        if (useMotionGate && motion < motionCutoff)
        {
            EmitScores(t0, 0f, 0f, false, motion);
            // still close any open event
            CloseEventIfNeeded(t0);
            return;
        }

        // 2) IMU → shared embedding (512) for arousal, and IMU embedding for fusion
        // TODO: call your real IMU encoder (on-device or network). For now, placeholder:
        var eImuShared = PlaceholderImuEmbedding(imuFlat);   // 512 L2-normed
        var eImu512 = eImuShared;                         // reuse for fusion

        // 3) MFCC → embedding (512). If mic off, pass zeros (already in your MFCC component if you choose).
        var eMfcc512 = PlaceholderMfccEmbedding(mfccFlat);   // 512

        // 4) C_social probability
        float p_social = PlaceholderClassifierProb(eImu512, eMfcc512);

        // 5) S_arousal score with anchors: cos(e, high) - cos(e, low)
        float s_arousal = 0f;
        if (eHigh != null && eLow != null)
            s_arousal = CosSim(eImuShared, eHigh) - CosSim(eImuShared, eLow);

        bool hase = (p_social >= thetaSocial) && (s_arousal >= thetaArousal);

        EmitScores(t0, p_social, s_arousal, hase, motion);
        Debounce(t0, hase);
    }

    // ---------- Helpers ----------
    static float MotionEnergy(float[] imu)
    {
        // imu is 800*6 [ax,ay,az,gx,gy,gz], compute std over time per axis → L2 across 6
        int n = imu.Length / 6;
        double[] mean = new double[6];
        for (int i = 0; i < n; i++)
            for (int c = 0; c < 6; c++) mean[c] += imu[i * 6 + c];
        for (int c = 0; c < 6; c++) mean[c] /= n;

        double[] v = new double[6];
        for (int i = 0; i < n; i++)
            for (int c = 0; c < 6; c++) { double d = imu[i * 6 + c] - mean[c]; v[c] += d * d; }
        double l2 = 0;
        for (int c = 0; c < 6; c++) l2 += v[c] / Math.Max(1, n - 1);
        return (float)Math.Sqrt(l2);
    }

    static float CosSim(float[] a, float[] b)
    {
        if (a == null || b == null || a.Length != b.Length) return 0f;
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        return (float)(dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-9));
    }

    // ===== PLACEHOLDERS (replace with real model calls) =====
    static float[] PlaceholderImuEmbedding(float[] imu)
    {
        // Stub: project down by simple averaging blocks → then L2 norm
        var e = new float[512];
        int stride = Math.Max(1, (imu.Length) / 512);
        for (int i = 0; i < 512; i++)
        {
            double s = 0; int start = i * stride, end = Math.Min(start + stride, imu.Length);
            for (int k = start; k < end; k++) s += imu[k];
            e[i] = (float)(s / Math.Max(1, end - start));
        }
        // L2 norm
        double n = 0; for (int i = 0; i < 512; i++) n += e[i] * e[i];
        n = Math.Sqrt(n) + 1e-9; for (int i = 0; i < 512; i++) e[i] /= (float)n;
        return e;
    }

    static float[] PlaceholderMfccEmbedding(float[] mfcc)
    {
        var e = new float[512];
        int stride = Math.Max(1, (mfcc.Length) / 512);
        for (int i = 0; i < 512; i++)
        {
            double s = 0; int start = i * stride, end = Math.Min(start + stride, mfcc.Length);
            for (int k = start; k < end; k++) s += mfcc[k];
            e[i] = (float)(s / Math.Max(1, end - start));
        }
        return e;
    }

    static float PlaceholderClassifierProb(float[] eImu, float[] eMfcc)
    {
        // Stub logistic on simple dot
        double dot = 0; int n = Math.Min(eImu.Length, eMfcc.Length);
        for (int i = 0; i < n; i++) dot += eImu[i] * eMfcc[i];
        double logit = 0.5 * dot; // scale
        return (float)(1.0 / (1.0 + Math.Exp(-logit)));
    }

    // ---------- Output / Debounce ----------
    void EmitScores(double t0, float p, float s, bool hase, float motion)
    {
        Debug.Log($"[SC] t0={t0:F0} p_social={p:F3} s_arousal={s:F3} hase={hase} motion={motion:F3}");
        if (hase && prompt) prompt.ShowOnce("Try a slow breath before your next sentence.");
    }

    void Debounce(double t0, bool hase)
    {
        if (!emitEvents) return;

        if (hase)
        {
            if (currentEvent == null)
                currentEvent = new EventAcc { start = t0, end = t0 + 20.0, n = 1 };
            else
            {
                // extend event
                currentEvent.end = t0 + 20.0;
                currentEvent.n++;
            }
        }
        else
        {
            // close if gap > merge window
            CloseEventIfNeeded(t0);
        }
    }

    void CloseEventIfNeeded(double t0)
    {
        if (currentEvent == null) return;
        if (t0 - currentEvent.end > mergeGapSecs)
        {
            Debug.Log($"[SC] EVENT id=auto start={currentEvent.start:F0} end={currentEvent.end:F0} dur={(currentEvent.end - currentEvent.start):F0}s n={currentEvent.n}");
            currentEvent = null;
        }
    }
}
