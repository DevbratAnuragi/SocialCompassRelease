using System.Collections.Generic;
using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class TongueTwisterPanel : MonoBehaviour
{
    [Header("UI")]
    public TMP_Text phraseText;
    public TMP_Text scoreText;
    public TMP_Text roundText;
    public Button nextButton;             // shows "Start" initially, then "Next"
    public Image timerFill;               // Image Type = Filled (Horizontal)

    [Header("Next Button Hop (horizontal)")]
    public RectTransform rightPanel;      // parent container for the button
    public float edgeMargin = 24f;
    public float hopDuration = 0.15f;
    RectTransform _btnRT;
    TMP_Text _nextBtnLabel;

    [Header("Flow")]
    [Tooltip("How many items to show in one run")]
    public int totalRounds = 10;

    [Tooltip("Seconds allowed for the first round")]
    public float startSeconds = 6f;

    [Tooltip("Seconds allowed for the last round (difficulty increases linearly)")]
    public float endSeconds = 2.5f;

    [Tooltip("Auto-start on enable (set FALSE for Start button UX)")]
    public bool autoStart = false;

    [Header("Timer Color")]
    public Color timerColorFull = new Color(0.12f, 0.74f, 0.23f); // green
    public Color timerColorMid = new Color(1.00f, 0.85f, 0.00f); // yellow
    public Color timerColorLow = new Color(0.86f, 0.12f, 0.12f); // red

    [Header("Content")]
    [TextArea(3, 10)]
    public List<string> tongueTwisters = new List<string>()
    {
        "Unique New York.",
        "Red leather, yellow leather.",
        "Six sticky skeletons.",
        "Top chopstick shops stock top chopsticks.",
        "Irish wristwatch, Swiss wristwatch.",
        "She sells seashells by the seashore.",
        "Which wristwatches are Swiss wristwatches?",
        "Six sleek swans swam swiftly southwards.",
        "Betty Botter bought a bit of butter.",
        "Peter Piper picked a peck of pickled peppers."
    };

    int _roundIndex = -1;   // -1 = not started
    int _score = 0;
    float _timeLeft = 0f;
    bool _running = false;  // timers/rounds active?
    Coroutine _hopAnim;

    void Awake()
    {
        if (nextButton)
        {
            nextButton.onClick.AddListener(OnNextPressed);
            _btnRT = nextButton.GetComponent<RectTransform>();
            _nextBtnLabel = nextButton.GetComponentInChildren<TMP_Text>();

            // Ensure center-bottom anchor so X jitter is easy.
            if (_btnRT)
            {
                _btnRT.anchorMin = new Vector2(0.5f, _btnRT.anchorMin.y);
                _btnRT.anchorMax = new Vector2(0.5f, _btnRT.anchorMax.y);
                _btnRT.pivot = new Vector2(0.5f, 0.5f);
            }
        }

        if (timerFill)
        {
            timerFill.type = Image.Type.Filled;
            timerFill.fillMethod = Image.FillMethod.Horizontal;
            timerFill.fillOrigin = (int)Image.OriginHorizontal.Left;
            timerFill.fillAmount = 0f;          // hidden/empty before start
            timerFill.color = timerColorFull;
        }

        // Initial UI text
        if (_nextBtnLabel) _nextBtnLabel.text = "Start";
        if (phraseText) phraseText.text = "Press Start to begin the tongue-twister challenge.";
        UpdateHUD();
    }

    void OnEnable()
    {
        if (autoStart) StartRun();
    }

    /// <summary>Call this if you want to start via inspector event instead of the Next/Start button.</summary>
    public void StartFromButton() => OnNextPressed();

    public void StartRun()
    {
        _score = 0;
        _roundIndex = -1;   // we advance immediately to 0 in NextRound()
        _running = true;
        NextRound();
        UpdateHUD();
        if (_nextBtnLabel) _nextBtnLabel.text = "Next";
    }

    void Update()
    {
        if (!_running) return;

        _timeLeft -= Time.deltaTime;

        if (timerFill)
        {
            float roundDur = Mathf.Max(0.0001f, RoundSeconds(_roundIndex));
            float frac = Mathf.Clamp01(_timeLeft / roundDur);
            timerFill.fillAmount = frac;

            // Color: green → yellow → red
            if (frac > 0.4f)
            {
                float t = Mathf.InverseLerp(1f, 0.4f, frac);
                timerFill.color = Color.Lerp(timerColorFull, timerColorMid, t);
            }
            else
            {
                float t = Mathf.InverseLerp(0.4f, 0f, frac);
                timerFill.color = Color.Lerp(timerColorMid, timerColorLow, t);
            }
        }

        if (_timeLeft <= 0f)
        {
            // time’s up → fail this round and advance
            Advance(failed: true);
        }
    }

    void OnNextPressed()
    {
        // If not started yet, treat this as "Start".
        if (!_running && _roundIndex < 0)
        {
            StartRun();
            return;
        }
        if (!_running) return; // finished state; wait for RestartFromButton()

        // Pressed before time ran out → success
        if (_timeLeft > 0f)
        {
            _score++;
            Advance(failed: false);
        }
        // else ignore; Update() will handle timeout.
    }

    void Advance(bool failed)
    {
        // Place for success/fail feedback if you want (sound/flash)
        NextRound();
        UpdateHUD();
    }

    void NextRound()
    {
        _roundIndex++;

        // Finished all rounds → show summary
        if (_roundIndex >= totalRounds)
        {
            EndRun();
            return;
        }

        // Set text
        string line = GetTwister(_roundIndex);
        if (phraseText) phraseText.text = line;

        // Reset timer
        _timeLeft = RoundSeconds(_roundIndex);
        if (timerFill)
        {
            timerFill.fillAmount = 1f;
            timerFill.color = timerColorFull;
        }

        // Hop the Next button horizontally
        PositionNextButtonRandomX();

        // Make sure the button says "Next" once we’ve started.
        if (_nextBtnLabel) _nextBtnLabel.text = "Next";
    }

    void EndRun()
    {
        _running = false;
        if (phraseText)
            phraseText.text = $"Done!\nScore: <b>{_score}/{totalRounds}</b>\nPress Start to try again.";
        if (timerFill)
        {
            timerFill.fillAmount = 0f;
            timerFill.color = timerColorFull;
        }

        // Reuse the button to restart
        if (nextButton)
        {
            nextButton.onClick.RemoveListener(OnNextPressed);
            nextButton.onClick.AddListener(RestartFromButton);
        }
        if (_nextBtnLabel) _nextBtnLabel.text = "Start";
    }

    void RestartFromButton()
    {
        if (nextButton)
        {
            nextButton.onClick.RemoveListener(RestartFromButton);
            nextButton.onClick.AddListener(OnNextPressed);
        }
        // Reset label to Next once started
        if (_nextBtnLabel) _nextBtnLabel.text = "Next";
        StartRun();
    }

    float RoundSeconds(int roundIdx)
    {
        // Linear interpolation from startSeconds → endSeconds across totalRounds
        if (totalRounds <= 1) return endSeconds;
        float t = Mathf.Clamp01(roundIdx / Mathf.Max(1f, (totalRounds - 1f)));
        return Mathf.Lerp(startSeconds, endSeconds, t);
    }

    string GetTwister(int roundIdx)
    {
        if (tongueTwisters == null || tongueTwisters.Count == 0)
            return "Add tongue twisters in the Inspector.";

        if (roundIdx < tongueTwisters.Count) return tongueTwisters[roundIdx];
        return tongueTwisters[roundIdx % tongueTwisters.Count];
    }

    void UpdateHUD()
    {
        if (scoreText) scoreText.text = $"Score: {_score}";
        int shownRound = Mathf.Clamp(_roundIndex + 1, 0, totalRounds);
        if (roundText) roundText.text = $"Round: {shownRound}/{totalRounds}";
    }

    // ─────────────────────────────────────────────────────────────────────
    // Button random horizontal position with small slide animation
    // ─────────────────────────────────────────────────────────────────────
    public void PositionNextButtonRandomX()
    {
        if (_btnRT == null || rightPanel == null) return;

        LayoutRebuilder.ForceRebuildLayoutImmediate(rightPanel);

        float panelW = rightPanel.rect.width;
        float btnW = _btnRT.rect.width;

        float halfSpan = 0.5f * panelW - 0.5f * btnW - edgeMargin;
        halfSpan = Mathf.Max(halfSpan, 0f);

        float targetX = Random.Range(-halfSpan, +halfSpan);

        if (_hopAnim != null) StopCoroutine(_hopAnim);
        _hopAnim = StartCoroutine(SlideX(_btnRT, targetX, hopDuration));
    }

    IEnumerator SlideX(RectTransform rt, float targetX, float dur)
    {
        float t = 0f;
        float startX = rt.anchoredPosition.x;
        dur = Mathf.Max(0.0001f, dur);

        while (t < dur)
        {
            t += Time.unscaledDeltaTime;
            float k = Mathf.SmoothStep(0f, 1f, t / dur);
            float x = Mathf.LerpUnclamped(startX, targetX, k);
            rt.anchoredPosition = new Vector2(x, rt.anchoredPosition.y);
            yield return null;
        }
        rt.anchoredPosition = new Vector2(targetX, rt.anchoredPosition.y);
    }
}
