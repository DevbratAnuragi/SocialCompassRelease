using UnityEngine;

/// <summary>
/// Provides a single, consistent, monotonic clock source for all data producers.
/// Uses Time.realtimeSinceStartupAsDouble, which is not affected by wall-clock changes or time scale.
/// </summary>
public static class MonoClock
{
    public static double Now() => Time.realtimeSinceStartupAsDouble;
}
