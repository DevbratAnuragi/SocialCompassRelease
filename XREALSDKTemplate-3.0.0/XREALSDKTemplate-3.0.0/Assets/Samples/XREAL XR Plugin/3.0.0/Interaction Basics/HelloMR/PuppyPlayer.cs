using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;
using UnityEngine.Video;

public class PuppyPlayer : MonoBehaviour
{
    [Header("UI")]
    public RawImage target;                 // drag your UI RawImage here
    public bool muteAudio = true;

    [Header("Local clip name in StreamingAssets")]
    public string streamingAssetsFile = "puppy.mp4";

    VideoPlayer _vp;
    RenderTexture _rt;
    string _localPath;                      // persistentDataPath/puppy.mp4

    void Awake()
    {
        if (!target) target = GetComponentInChildren<RawImage>(true);

        _vp = gameObject.AddComponent<VideoPlayer>();
        _vp.playOnAwake = false;
        _vp.isLooping = true;
        _vp.audioOutputMode = VideoAudioOutputMode.None;
        _vp.aspectRatio = VideoAspectRatio.FitInside;

        // modest RT is fine for a small UI chip; adjust if your video is larger
        _rt = new RenderTexture(512, 512, 0, RenderTextureFormat.ARGB32);
        _rt.Create();
        _vp.targetTexture = _rt;
        if (target) target.texture = _rt;

#if UNITY_ANDROID && !UNITY_EDITOR
        _localPath = Path.Combine(Application.persistentDataPath, streamingAssetsFile);
#else
        _localPath = Path.Combine(Application.streamingAssetsPath, streamingAssetsFile);
#endif
    }

    void OnDestroy()
    {
        if (_vp) { if (_vp.isPlaying) _vp.Stop(); _vp.targetTexture = null; }
        if (_rt) { _rt.Release(); Destroy(_rt); }
    }

    public void StopPuppy()
    {
        if (_vp && _vp.isPlaying) _vp.Stop();
    }

    public void PlayPuppy(string urlOverride = null)
    {
        StopAllCoroutines();
        StartCoroutine(PrepareAndPlay(urlOverride));
    }

    IEnumerator PrepareAndPlay(string urlOverride)
    {
        // Choose source: override URL (http/https) OR local file
        if (!string.IsNullOrEmpty(urlOverride))
        {
            _vp.source = VideoSource.Url;
            _vp.url = urlOverride;
        }
        else
        {
            yield return EnsureLocalFileReady();
            _vp.source = VideoSource.Url;
#if UNITY_ANDROID && !UNITY_EDITOR
            _vp.url = "file://" + _localPath; // explicit file:// helps on Android
#else
            _vp.url = _localPath;
#endif
        }

        _vp.Prepare();
        float t = 0f;
        while (!_vp.isPrepared && t < 3f) { t += Time.unscaledDeltaTime; yield return null; }
        if (_vp.isPrepared) _vp.Play();
    }

    IEnumerator EnsureLocalFileReady()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        if (File.Exists(_localPath)) yield break;

        // Read from StreamingAssets (jar) and write to persistentDataPath
        string src = Path.Combine(Application.streamingAssetsPath, streamingAssetsFile);
        using (var req = UnityWebRequest.Get(src))
        {
            yield return req.SendWebRequest();
#if UNITY_2020_2_OR_NEWER
            if (req.result != UnityWebRequest.Result.Success)
#else
            if (req.isNetworkError || req.isHttpError)
#endif
            {
                Debug.LogWarning("Puppy copy failed: " + req.error);
                yield break;
            }
            try
            {
                File.WriteAllBytes(_localPath, req.downloadHandler.data);
            }
            catch (System.Exception e)
            {
                Debug.LogWarning("Write video failed: " + e);
            }
        }
#else
        // non-Android: StreamingAssets is directly readable
        yield break;
#endif
    }
}
