using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

public class GifSpritePlayer : MonoBehaviour
{
    [Header("UI")]
    public Image target;                      // assign your UI Image

    [Header("Frames (manual)")]
    public List<Sprite> frames;               // optional: drag frames here

    [Header("Auto-load (Resources/)")]
    [Tooltip("If set, loads all Sprites from Resources/<resourcesPath>. " +
             "Works with a sliced spritesheet or multiple sprite assets in that folder.")]
    public string resourcesPath;              // e.g. "PuppyA" or "Puppies/Set1"
    public bool autoLoadFromResources = false;
    public bool sortByName = true;            // sorts frames by sprite.name
    public bool reverseOrder = false;         // flip order after sort

    [Header("Playback")]
    [Range(1, 60)] public int fps = 12;
    public bool loop = true;
    public bool autoPlay = true;

    int _i;
    float _accum;
    bool _playing;

    void Awake()
    {
        // If requested, load frames from Resources on startup
        if (autoLoadFromResources && !string.IsNullOrEmpty(resourcesPath))
        {
            // Loads:
            //  - All sub-sprites from a sliced spritesheet at Resources/<resourcesPath>
            //  - All sprite assets under a folder Resources/<resourcesPath>
            var loaded = Resources.LoadAll<Sprite>(resourcesPath);
            if (loaded != null && loaded.Length > 0)
            {
                if (sortByName) loaded = loaded.OrderBy(s => s.name).ToArray();
                if (reverseOrder) loaded = loaded.Reverse().ToArray();
                frames = loaded.ToList();
            }
            else
            {
                Debug.LogWarning($"[GifSpritePlayer] No sprites found at Resources/{resourcesPath}");
            }
        }

        if (!target) target = GetComponent<Image>();
    }

    void OnEnable()
    {
        if (autoPlay) Play();
    }

    void Update()
    {
        if (!_playing || target == null || frames == null || frames.Count == 0) return;

        _accum += Time.unscaledDeltaTime;
        float frameDur = 1f / Mathf.Max(1, fps);

        while (_accum >= frameDur)
        {
            _accum -= frameDur;
            _i++;

            if (_i >= frames.Count)
            {
                if (loop) _i = 0;
                else { Stop(); return; }
            }
            target.sprite = frames[_i];
        }
    }

    public void Play()
    {
        if (target == null)
        {
            Debug.LogWarning("[GifSpritePlayer] No target Image assigned.");
            return;
        }
        if (frames == null || frames.Count == 0)
        {
            Debug.LogWarning("[GifSpritePlayer] No frames assigned or loaded.");
            return;
        }
        _i = 0;
        _accum = 0f;
        target.sprite = frames[_i];
        _playing = true;
        target.SetVerticesDirty(); // refresh UI
    }

    public void Stop() => _playing = false;
}
