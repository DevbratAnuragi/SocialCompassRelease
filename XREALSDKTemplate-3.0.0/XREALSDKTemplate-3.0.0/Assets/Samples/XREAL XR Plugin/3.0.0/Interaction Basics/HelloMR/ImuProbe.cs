using UnityEngine;

public class ImuProbe : MonoBehaviour
{
    void Start()
    {
        Debug.Log($"[PROBE] supportsAccelerometer={SystemInfo.supportsAccelerometer} supportsGyro={SystemInfo.supportsGyroscope}");
        Input.gyro.enabled = true;
    }
    void Update()
    {
        var a = Input.acceleration;
        var g = Input.gyro.rotationRateUnbiased;
        if (Time.frameCount % 30 == 0)
            Debug.Log($"[PROBE] acc=({a.x:F3},{a.y:F3},{a.z:F3}) gyro=({g.x:F3},{g.y:F3},{g.z:F3})");
    }
}
