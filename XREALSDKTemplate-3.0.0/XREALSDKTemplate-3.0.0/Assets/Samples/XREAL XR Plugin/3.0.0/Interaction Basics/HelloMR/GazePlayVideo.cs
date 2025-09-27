using UnityEngine;
using UnityEngine.Video;

public class GazePlayVideo : MonoBehaviour
{
    [Header("Refs")]
    public Camera xrCamera;           // assign your XR/HMD camera
    public Transform target;          // the panel/board to look at (usually the RawImage's transform)
    public VideoPlayer player;        // the VideoPlayer component
    public AudioSource audioSource;   // optional; if using audio

    [Header("Gaze Settings")]
    [Tooltip("Max angle off the forward gaze to count as 'looking' (degrees).  e.g., 20–25°")]
    public float viewAngleDeg = 25f;
    [Tooltip("Only consider gaze if within this distance (meters). Set 0 to ignore distance.")]
    public float maxDistance = 0f;
    [Tooltip("Min seconds the user must keep looking before play toggles (prevents flicker).")]
    public float dwellSeconds = 0.15f;

    [Header("Behavior")]
    public bool pauseWhenNotLooked = true;     // pause instead of stop (keeps frame)
    public bool unmuteOnLook = true;

    float _lookTimer = 0f;
    bool _isLooking = false;

    void Awake()
    {
        if (!xrCamera) xrCamera = Camera.main;
        if (!player) player = GetComponent<VideoPlayer>();
        if (player && audioSource)
        {
            player.audioOutputMode = VideoAudioOutputMode.AudioSource;
            if (player.audioTrackCount > 0) player.SetTargetAudioSource(0, audioSource);
        }
    }

    void Update()
    {
        if (!xrCamera || !target || !player) return;

        // Vector from head to panel (yaw-only facing is OK; we use full 3D)
        Vector3 toTarget = (target.position - xrCamera.transform.position);
        float dist = toTarget.magnitude;
        if (maxDistance > 0f && dist > maxDistance)
        {
            UpdateLook(false);
            return;
        }

        toTarget.Normalize();
        // cosine of angle between camera forward and target direction
        float dot = Vector3.Dot(xrCamera.transform.forward, toTarget);
        // Convert threshold to cosine
        float cosThresh = Mathf.Cos(viewAngleDeg * Mathf.Deg2Rad);
        bool lookingNow = dot >= cosThresh;

        UpdateLook(lookingNow);
    }

    void UpdateLook(bool lookingNow)
    {
        if (lookingNow)
        {
            _lookTimer += Time.deltaTime;
            if (!_isLooking && _lookTimer >= dwellSeconds)
            {
                _isLooking = true;
                TogglePlayback(true);
            }
        }
        else
        {
            _lookTimer = 0f;
            if (_isLooking)
            {
                _isLooking = false;
                TogglePlayback(false);
            }
        }
    }

    void TogglePlayback(bool shouldPlay)
    {
        if (shouldPlay)
        {
            if (unmuteOnLook && audioSource) audioSource.mute = false;
            if (!player.isPlaying) player.Play();
        }
        else
        {
            if (pauseWhenNotLooked) player.Pause();
            else player.Stop();

            if (unmuteOnLook && audioSource) audioSource.mute = true;
        }
    }
}
