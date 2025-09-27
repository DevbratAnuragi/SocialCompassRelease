// GlassesImu40Hz.cs
// Collects HMD (head) linear acceleration (m/s^2) and angular velocity (rad/s)
// via Unity XR at ~40 Hz, packs 20 s windows => float32 [800,6] in [ax,ay,az,gx,gy,gz].

using System;
using System.Collections;
using UnityEngine;
using UnityEngine.XR; // XR InputDevices

public class GlassesImu40Hz : MonoBehaviour
{
    [Range(30, 100)] public int targetHz = 40;
    [Tooltip("20 s => 800 samples @ 40 Hz")] public int windowSeconds = 20;
    [Tooltip("Stride seconds; 20 = non-overlap, 10 = 50% overlap")]
    public int strideSeconds = 20;

    public event Action<float[], double> OnWindowReady; // (imu6, epoch_start_sec)

    float _dt;
    float[] _buf; int _n;
    double _epochStart;
    Coroutine _loop;

    // XR head device
    InputDevice _head;

    void OnEnable()
    {
        _dt = 1f / Mathf.Max(1, targetHz);
        int N = Mathf.RoundToInt(windowSeconds / _dt);
        _buf = new float[N * 6];
        _n = 0;
        _epochStart = Now();

        // Try to fetch the head device
        _head = InputDevices.GetDeviceAtXRNode(XRNode.Head);

        // if device isn't valid yet, wait for XR to initialize
        StartCoroutine(WaitForHeadThenRun());
    }

    IEnumerator WaitForHeadThenRun()
    {
        var timeout = Time.realtimeSinceStartup + 5f;
        while ((!_head.isValid) && Time.realtimeSinceStartup < timeout)
        {
            _head = InputDevices.GetDeviceAtXRNode(XRNode.Head);
            yield return null;
        }
        _loop = StartCoroutine(SampleLoop());
    }

    void OnDisable()
    {
        if (_loop != null) StopCoroutine(_loop);
    }

    IEnumerator SampleLoop()
    {
        var wait = new WaitForSecondsRealtime(_dt);
        double strideStart = _epochStart;

        while (true)
        {
            Vector3 accel = Vector3.zero;      // m/s^2
            Vector3 angVel = Vector3.zero;     // rad/s

            // Some XR providers expose these directly; if not, values remain zero.
            // (Commonly available on modern XR stacks; if missing, you can derive
            //  from pose deltas as a fallback.)
            _head.TryGetFeatureValue(CommonUsages.deviceAcceleration, out accel);
            _head.TryGetFeatureValue(CommonUsages.deviceAngularVelocity, out angVel);

            int i = _n * 6;
            _buf[i + 0] = accel.x; _buf[i + 1] = accel.y; _buf[i + 2] = accel.z;
            _buf[i + 3] = angVel.x; _buf[i + 4] = angVel.y; _buf[i + 5] = angVel.z;
            _n++;

            if (_n * _dt >= windowSeconds - 1e-4f)
            {
                var copy = new float[_buf.Length];
                Buffer.BlockCopy(_buf, 0, copy, 0, sizeof(float) * _buf.Length);
                OnWindowReady?.Invoke(copy, _epochStart);

                strideStart += strideSeconds;
                _epochStart = strideStart;
                _n = 0;
            }
            yield return wait;
        }
    }

    static double Now() => DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0;
}
