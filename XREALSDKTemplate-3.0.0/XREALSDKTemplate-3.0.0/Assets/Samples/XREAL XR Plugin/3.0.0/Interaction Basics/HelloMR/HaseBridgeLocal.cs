using System;
using System.IO;
using UnityEngine;

public class HaseBridgeLocal : MonoBehaviour
{
    [Header("Wiring")]
    public WindowMux mux;           // assign WindowMux from the scene
    public PromptSlide prompt;      // assign your slide panel

    [Header("Behavior")]
    [Tooltip("Show a prompt for every window (good for testing).")]
    public bool showPromptEachWindow = true;

    [Header("Optional saving (for offline checks)")]
    public bool saveWindows = false;
    public string subfolder = "hase_windows"; // under persistentDataPath

    void OnEnable()
    {
        if (mux) mux.OnHaseWindow += Handle;
    }
    void OnDisable()
    {
        if (mux) mux.OnHaseWindow -= Handle;
    }

    void Handle(float[] imu, float[] mfcc, double epochStart)
    {
        // imu: 800*6 floats, mfcc: 430*13 floats
        Debug.Log($"[HASE] Window @ {epochStart:F3}s  IMU {imu.Length}  MFCC {mfcc.Length}");

        if (saveWindows) SavePair(imu, mfcc, epochStart);

        if (showPromptEachWindow && prompt)
        {
            // Hardcoded “LLM-style” line for now
            prompt.ShowOnce("Try a slow breath before your next sentence.");
        }

        // If you want a super-light “detector” placeholder:
        // float aStd = EstimateAccelStd(imu);
        // if (aStd > 1.5f) prompt.ShowOnce("Consider a brief pause to reset your pace.");
    }

    static float EstimateAccelStd(float[] imu)
    {
        // imu layout: [ax,ay,az,gx,gy,gz] repeating
        int n = imu.Length / 6;
        double m = 0, v = 0;
        for (int i = 0; i < n; i++)
        {
            int k = i * 6;
            double a = Math.Sqrt(imu[k + 0] * imu[k + 0] + imu[k + 1] * imu[k + 1] + imu[k + 2] * imu[k + 2]); // m/s^2
            double d = a - m;
            m += d / (i + 1);
            v += d * (a - m);
        }
        return (float)Math.Sqrt(v / Math.Max(1, n - 1));
    }

    void SavePair(float[] imu, float[] mfcc, double t0)
    {
        try
        {
            string root = Path.Combine(Application.persistentDataPath, subfolder);
            Directory.CreateDirectory(root);
            string stamp = Math.Floor(t0).ToString();
            string imuBin = Path.Combine(root, $"{stamp}_imu.bin");
            string mfccBin = Path.Combine(root, $"{stamp}_mfcc.bin");
            string meta = Path.Combine(root, $"{stamp}_meta.json");

            using (var fs = new FileStream(imuBin, FileMode.Create, FileAccess.Write))
            using (var bw = new BinaryWriter(fs))
                foreach (var f in imu) bw.Write(f);   // little-endian float32

            using (var fs = new FileStream(mfccBin, FileMode.Create, FileAccess.Write))
            using (var bw = new BinaryWriter(fs))
                foreach (var f in mfcc) bw.Write(f);

            File.WriteAllText(meta,
                $"{{\"epoch_start_s\":{t0:F3},\"imu_shape\":[800,6],\"mfcc_shape\":[430,13],\"units\":\"m/s^2,rad/s; MFCC dims=13\",\"sr\":16000}}");

            Debug.Log($"[HASE] Saved {imuBin} / {mfccBin}");
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[HASE] Save failed: {e}");
        }
    }
}
