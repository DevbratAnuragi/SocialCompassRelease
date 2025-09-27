using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.Collections;

public class IntroPanelController : MonoBehaviour
{
    [Header("Wiring")]
    public CanvasGroup cg;                 // on Panel_Intro
    public TMP_Text titleText;
    public TMP_Text taglineText;
    public TMP_Text bodyText;
    public TMP_Text bulletsText;
    public Button btnStart;
    public Button btnSkip;

    [Header("Panels To Show After Intro")]
    public GameObject leftPanelRoot;       // video panel root
    public GameObject rightPanelRoot;      // tongue-twister panel root

    [Header("Behavior")]
    public float fadeSeconds = 0.35f;
    public bool showOnStart = true;
    public bool pauseWorldBehind = true;   // optional: set Time.timeScale

    // Optional: trigger a first subtle prompt when starting
    public PromptSlide promptSlide;
    [TextArea] public string firstPrompt = "Try a slow, even breath before you begin.";

    void Awake()
    {
        if (!cg) cg = GetComponent<CanvasGroup>();
        if (btnStart) btnStart.onClick.AddListener(OnStart);
        if (btnSkip) btnSkip.onClick.AddListener(OnSkip);

        if (leftPanelRoot) leftPanelRoot.SetActive(false);
        if (rightPanelRoot) rightPanelRoot.SetActive(false);

        if (showOnStart) Show();
        else HideImmediate();
    }

    public void Show()
    {
        gameObject.SetActive(true);
        if (pauseWorldBehind) Time.timeScale = 0f;
        StartCoroutine(FadeTo(1f, fadeSeconds));
    }

    public void Hide()
    {
        StartCoroutine(FadeOutAndDisable());
    }

    void HideImmediate()
    {
        if (cg)
        {
            cg.alpha = 0f;
            cg.interactable = false;
            cg.blocksRaycasts = false;
        }
        gameObject.SetActive(false);
    }

    void OnStart()
    {
        // Enable your left/right panels
        if (leftPanelRoot) leftPanelRoot.SetActive(true);
        if (rightPanelRoot) rightPanelRoot.SetActive(true);

        // Resume time
        if (pauseWorldBehind) Time.timeScale = 1f;

        // Optional: nudge user with the first prompt
        if (promptSlide && !string.IsNullOrEmpty(firstPrompt))
            promptSlide.ShowOnce(firstPrompt);

        Hide();
    }

    void OnSkip()
    {
        // Same as Start, but you can omit the first prompt if you prefer
        if (leftPanelRoot) leftPanelRoot.SetActive(true);
        if (rightPanelRoot) rightPanelRoot.SetActive(true);
        if (pauseWorldBehind) Time.timeScale = 1f;
        Hide();
    }

    IEnumerator FadeOutAndDisable()
    {
        yield return FadeTo(0f, fadeSeconds);
        cg.interactable = false;
        cg.blocksRaycasts = false;
        gameObject.SetActive(false);
    }

    IEnumerator FadeTo(float a1, float dur)
    {
        cg.blocksRaycasts = true;
        cg.interactable = (a1 > 0.5f);

        float a0 = cg.alpha;
        float t = 0f;
        // Use unscaled time to work during pause
        while (t < dur)
        {
            t += Time.unscaledDeltaTime;
            cg.alpha = Mathf.Lerp(a0, a1, Mathf.SmoothStep(0f, 1f, t / dur));
            yield return null;
        }
        cg.alpha = a1;
        cg.interactable = (a1 > 0.5f);
        cg.blocksRaycasts = (a1 > 0f);
    }

    // Optional: keyboard shortcut to re-open intro
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.I))
        {
            if (!gameObject.activeSelf) Show();
        }
    }
}
