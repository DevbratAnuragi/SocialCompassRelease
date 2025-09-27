using UnityEngine;
using System.Collections;

/// <summary>
/// Make a world-space canvas (or any object) move toward the viewer
/// when looked at. Movement accelerates as distance decreases.
/// When it reaches a stop distance, it "resets" to start for looping.
/// </summary>
[RequireComponent(typeof(RectTransform))]
public class CanvasStalker : MonoBehaviour
{
    [Header("Target (usually the HMD/Main Camera)")]
    public Transform viewer;                    // assign Camera.main if null at Start

    [Header("Look Activation")]
    [Range(-1f, 1f)] public float lookDotThreshold = 0.8f; // cos(angle); 0.8 ~ 36°
    public bool requireInViewFrustum = true;               // optional frustum test

    [Header("Motion")]
    public float baseSpeed = 10.55f;             // m/s when far
    public float closeBoost = 3.5f;             // added speed scaled by 1/(distance+eps)
    public float minDistance = 0.6f;            // stop distance (trigger "scare")
    public float faceLerp = 12f;                // billboard turn rate toward viewer
    public float gravityLikePull = 5f;          // extra pull (0 = off)

    [Header("Loop / Reset")]
    public bool loop = true;
    public float resetDelay = 0.75f;            // pause before reset (for dramatic effect)
    public Vector3 resetJitter = new Vector3(0.05f, 0.05f, 0.05f); // tiny randomness

    [Header("FX (optional)")]
    public AudioSource whoosh;                  // loop/one-shot while moving
    public AudioClip scareClip;                 // play when reaching minDistance
    public bool onlyPlayWhooshWhileMoving = true;

    RectTransform _rt;
    Vector3 _startPos;
    Quaternion _startRot;
    bool _resetting;

    void Awake()
    {
       

        _startPos = transform.position;
        _startRot = transform.rotation;
    }

    void Start()
    {
        if (viewer == null && Camera.main != null) viewer = Camera.main.transform;
        if (whoosh != null && !whoosh.playOnAwake && !onlyPlayWhooshWhileMoving)
            whoosh.Play();
    }

    void Update()
    {
        if (viewer == null || _resetting) return;

        // Always softly face the viewer (billboard)
        var toCam = (viewer.position - transform.position);
        if (toCam.sqrMagnitude > 0.0001f)
        {
            var lookRot = Quaternion.LookRotation(toCam.normalized, Vector3.up);
            transform.rotation = Quaternion.Slerp(transform.rotation, lookRot, 1f - Mathf.Exp(-faceLerp * Time.deltaTime));
        }

        // Check "look at me"
        bool looking = IsViewerLookingAtMe();

        // Move only when being looked at
        if (looking)
        {
            float dist = Mathf.Max(0.001f, toCam.magnitude);
            // Speed increases as distance decreases
            float speed = baseSpeed + closeBoost / (dist + 0.0001f);
            if (gravityLikePull > 0f) speed += gravityLikePull * Time.deltaTime;

            Vector3 step = toCam.normalized * speed * Time.deltaTime;
            transform.position += step;

            // audio whoosh logic
            if (whoosh)
            {
                if (onlyPlayWhooshWhileMoving)
                {
                    if (!whoosh.isPlaying) whoosh.Play();
                }
                else if (!whoosh.isPlaying) whoosh.Play(); // ensure it's on
            }

            // Reached the scare distance?
            if (dist <= minDistance)
            {
                if (scareClip != null)
                    AudioSource.PlayClipAtPoint(scareClip, transform.position, 1f);

                if (loop) StartCoroutine(ResetAfterDelay());
                else enabled = false; // stop forever
            }
        }
        else
        {
            // stop whoosh when not moving (if configured)
            if (whoosh && onlyPlayWhooshWhileMoving && whoosh.isPlaying)
                whoosh.Pause();
        }
    }

    bool IsViewerLookingAtMe()
    {
        if (viewer == null) return false;

        Vector3 toObj = (transform.position - viewer.position).normalized;
        float dot = Vector3.Dot(viewer.forward, toObj);
        if (dot < lookDotThreshold) return false;

        if (!requireInViewFrustum) return true;

        // Rough frustum check (no alloc): project to viewport space
        Camera cam = Camera.main;
        if (cam == null) return true; // fallback

        Vector3 vp = cam.WorldToViewportPoint(transform.position);
        return (vp.z > 0f && vp.x >= 0f && vp.x <= 1f && vp.y >= 0f && vp.y <= 1f);
    }

    IEnumerator ResetAfterDelay()
    {
        _resetting = true;
        if (whoosh && onlyPlayWhooshWhileMoving && whoosh.isPlaying)
            whoosh.Pause();

        yield return new WaitForSeconds(resetDelay);

        // Reset back to start with tiny random jitter so it’s not identical each loop
        Vector3 j = new Vector3(
            Random.Range(-resetJitter.x, resetJitter.x),
            Random.Range(-resetJitter.y, resetJitter.y),
            Random.Range(-resetJitter.z, resetJitter.z)
        );
        transform.position = _startPos + j;
        transform.rotation = _startRot;
        _resetting = false;
    }

#if UNITY_EDITOR
    void OnDrawGizmosSelected()
    {
        Gizmos.color = Color.red;
        Gizmos.DrawWireSphere(transform.position, minDistance);
    }
#endif
}
