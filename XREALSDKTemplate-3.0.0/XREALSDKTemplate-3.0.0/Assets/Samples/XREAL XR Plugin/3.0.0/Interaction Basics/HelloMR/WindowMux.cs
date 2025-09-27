using UnityEngine;

public class WindowMux : MonoBehaviour
{
    public BeamImu40Hz imu;
    public new AudioMfccWindow audio;

    float[] _lastImu; double _tImu;
    float[] _lastMfcc; double _tMfcc;

    public delegate void HaseWindow(float[] imu800x6, float[] mfcc430x13, double epochStart);
    public event HaseWindow OnHaseWindow;

    void OnEnable()
    {
        imu.OnWindowReady += OnImu;
        audio.OnMfccWindowReady += OnMfcc;
    }
    void OnDisable()
    {
        imu.OnWindowReady -= OnImu;
        audio.OnMfccWindowReady -= OnMfcc;
    }

    void OnImu(float[] w, double t) { _lastImu = w; _tImu = t; TryEmit(); }
    void OnMfcc(float[] w, double t) { _lastMfcc = w; _tMfcc = t; TryEmit(); }
    double? _offset = null;   // calibration offset between IMU and audio
    void TryEmit()
    {
        if (_lastImu == null || _lastMfcc == null) return;

        // On first valid pair, lock in the offset
        if (_offset == null)
        {
            _offset = _tImu - _tMfcc;
            Debug.Log($"[MUX] Calibrated offset = {_offset:F3}s (IMU - Audio)");
        }

        // Apply the offset to audio timestamps so they line up with IMU
        double adjMfccTime = _tMfcc + _offset.Value;
        var skew = Mathf.Abs((float)(_tImu - adjMfccTime));

        Debug.Log($"[MUX] Emit skew={skew:F3}s imu={_tImu:F3} mfccAdj={adjMfccTime:F3}");

        // Use midpoint of adjusted times
        OnHaseWindow?.Invoke(_lastImu, _lastMfcc, _tImu);

        _lastImu = null;
        _lastMfcc = null;
    }
}
