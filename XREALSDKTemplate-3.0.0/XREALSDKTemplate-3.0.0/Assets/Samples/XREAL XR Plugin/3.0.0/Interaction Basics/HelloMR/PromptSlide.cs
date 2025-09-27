using System.Collections;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class PromptSlide : MonoBehaviour
{
    [Header("Wiring")]
    public RectTransform panel;     // assign your Panel_Prompt
    public TMP_Text promptText; // assign PromptText (TMP)

    [Header("Behavior")]
    public float slideDuration = 0.30f;
    public float holdSeconds = 3.0f;
    public float leftMargin = 24f;

    [Header("Optional media")]
    public GifSpritePlayer gif;   // assign if using PNG frames
    public PuppyPlayer puppy; // assign if using MP4

    Vector2 _offPos, _onPos;
    Coroutine _anim;

    void Awake()
    {
        if (!panel) panel = GetComponent<RectTransform>();
        LayoutRebuilder.ForceRebuildLayoutImmediate(panel);

        float width = panel.rect.width;
        _offPos = new Vector2(-width - 40f, panel.anchoredPosition.y);
        _onPos = new Vector2(leftMargin, panel.anchoredPosition.y);

        panel.anchoredPosition = _offPos;

        var cg = panel.GetComponent<CanvasGroup>();
        if (!cg) cg = panel.gameObject.AddComponent<CanvasGroup>();
        cg.alpha = 0f;
    }

    public void ShowOnce(string text, string optionalVideoUrl = null)
    {
        Debug.Log($"[PromptSlide] ShowOnce called with text: '{text}' at time: {Time.time}");
        if (promptText) promptText.text = text;
        if (_anim != null) StopCoroutine(_anim);
        _anim = StartCoroutine(ShowRoutine(optionalVideoUrl));
    }

    IEnumerator ShowRoutine(string videoUrl)
    {
        Debug.Log("[PromptSlide] --- ShowRoutine started. ---");
        var cg = panel.GetComponent<CanvasGroup>();

        // slide in + fade
        yield return Slide(panel, _offPos, _onPos, slideDuration, cg, 0f, 0.9f);

        // start media (whichever is assigned)
        if (gif) gif.Play();
        if (puppy) puppy.PlayPuppy(videoUrl);

        // hold
        yield return new WaitForSecondsRealtime(holdSeconds);

        // stop media
        if (gif) gif.Stop();
        if (puppy) puppy.StopPuppy();

        // slide out + fade
        yield return Slide(panel, _onPos, _offPos, 0.25f, cg, 0.9f, 0f);
        Debug.Log("[PromptSlide] --- ShowRoutine finished successfully. ---");
    }

    static IEnumerator Slide(RectTransform rt, Vector2 from, Vector2 to, float dur, CanvasGroup cg, float a0, float a1)
    {
        float t = 0f;
        while (t < dur)
        {
            t += Time.unscaledDeltaTime;
            float k = Mathf.SmoothStep(0f, 1f, Mathf.Clamp01(t / dur));
            rt.anchoredPosition = Vector2.LerpUnclamped(from, to, k);
            if (cg) cg.alpha = Mathf.Lerp(a0, a1, k);
            yield return null;
        }
        rt.anchoredPosition = to;
        if (cg) cg.alpha = a1;
    }
}
