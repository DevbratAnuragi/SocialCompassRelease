using System;
using System.Collections;
using UnityEngine;

public class BeamImu40Hz : MonoBehaviour
{
    [Range(30, 100)] public int targetHz = 40;
    public int windowSeconds = 20;
    public int strideSeconds = 20;
    public event Action<float[], double> OnWindowReady; // float[800*6], t0=start (epoch)

    // shared state written by sensor callbacks
    volatile float _ax, _ay, _az, _gx, _gy, _gz;

    float[] _raw;
    int _n;
    float _dt;
    Coroutine _loop;
    double _nextStartMono;

    // Android pieces
#if UNITY_ANDROID && !UNITY_EDITOR
    AndroidJavaObject _sensorMgr, _accel, _gyro;
    ImuListener _listener;
    bool _androidCallbacks;
#endif
    bool _useUnityFallback;

    void OnEnable()
    {
        _dt = 1f / Mathf.Max(1, targetHz);
        int rawN = Mathf.RoundToInt(windowSeconds / _dt);
        _raw = new float[rawN * 6];
        _n = 0;

        // align to stride on a monotonic clock
        double stride = strideSeconds;
        double now = Time.realtimeSinceStartupAsDouble;
        _nextStartMono = Math.Floor(now / stride) * stride;

#if UNITY_ANDROID && !UNITY_EDITOR
        try
        {
            var unity = new AndroidJavaClass("com.unity3d.player.UnityPlayer");
            var act = unity.GetStatic<AndroidJavaObject>("currentActivity");

            // Context.SENSOR_SERVICE is literally the string "sensor"
            _sensorMgr = act.Call<AndroidJavaObject>("getSystemService", "sensor");
            var sensorClass = new AndroidJavaClass("android.hardware.Sensor");

            int TYPE_ACCEL = sensorClass.GetStatic<int>("TYPE_ACCELEROMETER");
            int TYPE_GYRO  = sensorClass.GetStatic<int>("TYPE_GYROSCOPE");

            _accel = _sensorMgr.Call<AndroidJavaObject>("getDefaultSensor", TYPE_ACCEL);
            _gyro  = _sensorMgr.Call<AndroidJavaObject>("getDefaultSensor", TYPE_GYRO);

            Debug.Log($"[IMU] accel={(_accel!=null)} gyro={(_gyro!=null)}");

            _listener = new ImuListener(OnSensor);
            const int us40Hz = 25_000; // 40 Hz
            if (_accel != null) _sensorMgr.Call<bool>("registerListener", _listener, _accel, us40Hz);
            if (_gyro  != null) _sensorMgr.Call<bool>("registerListener", _listener, _gyro,  us40Hz);

            // detect if callbacks actually arrive; fallback if not
            StartCoroutine(EnsureCallbacksOrFallback());
        }
        catch (Exception e)
        {
            Debug.LogWarning("[IMU] Android sensor setup failed: " + e.Message);
            _useUnityFallback = true;
        }
#else
        _useUnityFallback = true;
#endif
        _loop = StartCoroutine(SampleLoop());
    }

    void OnDisable()
    {
        if (_loop != null) StopCoroutine(_loop);
#if UNITY_ANDROID && !UNITY_EDITOR
        try { if (_sensorMgr != null && _listener != null) _sensorMgr.Call("unregisterListener", _listener); }
        catch {}
#endif
    }

#if UNITY_ANDROID && !UNITY_EDITOR
    IEnumerator EnsureCallbacksOrFallback()
    {
        float t0 = Time.realtimeSinceStartup;
        float lastSum = _ax+_ay+_az+_gx+_gy+_gz;
        yield return new WaitForSeconds(1.0f); // give Android a second to fire events
        float nowSum = _ax+_ay+_az+_gx+_gy+_gz;
        _androidCallbacks = !Mathf.Approximately(lastSum, nowSum);
        if (!_androidCallbacks)
        {
            Debug.LogWarning("[IMU] No Android sensor callbacks after 1s -> using Unity fallback");
            _useUnityFallback = true;
        }
        else
        {
            Debug.Log("[IMU] Android sensor callbacks active");
        }
    }

    class ImuListener : AndroidJavaProxy
    {
        readonly Action<int, float[]> _cb;
        readonly AndroidJavaClass _sensorClass = new AndroidJavaClass("android.hardware.Sensor");
        readonly int _TYPE_ACCEL, _TYPE_GYRO;

        public ImuListener(Action<int, float[]> cb)
            : base("android.hardware.SensorEventListener")
        {
            _cb = cb;
            _TYPE_ACCEL = _sensorClass.GetStatic<int>("TYPE_ACCELEROMETER");
            _TYPE_GYRO  = _sensorClass.GetStatic<int>("TYPE_GYROSCOPE");
        }

        // void onSensorChanged(SensorEvent event)
        void onSensorChanged(AndroidJavaObject sensorEvent)
        {
            try
            {
                int type = sensorEvent.Get<AndroidJavaObject>("sensor").Get<int>("type");
                float[] values = AndroidJNIHelper.ConvertFromJNIArray<float[]>(
                    sensorEvent.Get<AndroidJavaObject>("values").GetRawObject());
                if (values != null && values.Length >= 3)
                    _cb?.Invoke(type, values);
            }
            catch (Exception e)
            {
                Debug.LogWarning("[IMU] onSensorChanged error: " + e.Message);
            }
        }

        // void onAccuracyChanged(Sensor sensor, int accuracy)
        void onAccuracyChanged(AndroidJavaObject sensor, int accuracy) {}
    }
#endif

    void OnSensor(int type, float[] v)
    {
        // TYPE_ACCELEROMETER == 1, TYPE_GYROSCOPE == 4
        if (type == 1) { _ax = v[0]; _ay = v[1]; _az = v[2]; }     // m/s^2
        else if (type == 4) { _gx = v[0]; _gy = v[1]; _gz = v[2]; } // rad/s
    }

    IEnumerator SampleLoop()
    {
        if (_useUnityFallback)
        {
            Input.gyro.enabled = true;
            Debug.Log("[IMU] Using Unity fallback (Input.acceleration/Input.gyro)");
        }

        var wait = new WaitForSecondsRealtime(_dt);
        while (true)
        {
            if (_useUnityFallback)
            {
                var a = Input.acceleration; // m/s^2 on Android
                var g = Input.gyro.rotationRateUnbiased; // rad/s
                _ax = a.x; _ay = a.y; _az = a.z;
                _gx = g.x; _gy = g.y; _gz = g.z;

               
            }

            int i = _n * 6;
            _raw[i + 0] = _ax; _raw[i + 1] = _ay; _raw[i + 2] = _az;
            _raw[i + 3] = _gx; _raw[i + 4] = _gy; _raw[i + 5] = _gz;
            _n++;

            if (_n * _dt >= windowSeconds - 1e-4f)
            {
                var imu800x6 = ResampleImuTo800(_raw, _n, 6);

                double t0_epoch = UnixFromMono(_nextStartMono); // window START
                OnWindowReady?.Invoke(imu800x6, t0_epoch);

                _nextStartMono += strideSeconds;
                _n = 0;
            }
            yield return wait;
        }
    }

    static float[] ResampleImuTo800(float[] src, int validN, int ch)
    {
        int tgtN = 800;
        var dst = new float[tgtN * ch];
        if (validN <= 1) return dst;
        for (int k = 0; k < tgtN; k++)
        {
            double t = (double)k * (validN - 1) / (tgtN - 1);
            int i0 = (int)Math.Floor(t);
            int i1 = Math.Min(i0 + 1, validN - 1);
            double a = t - i0;
            for (int c = 0; c < ch; c++)
            {
                float v0 = src[i0 * ch + c];
                float v1 = src[i1 * ch + c];
                dst[k * ch + c] = (float)((1 - a) * v0 + a * v1);
            }
        }
        return dst;
    }

    static double UnixFromMono(double monoSeconds)
    {
        // cache anchors
        if (_mono0 < 0)
        {
            _mono0 = Time.realtimeSinceStartupAsDouble;
            _unix0 = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0;
        }
        return _unix0 + (monoSeconds - _mono0);
    }
    static double _mono0 = -1, _unix0;
}
