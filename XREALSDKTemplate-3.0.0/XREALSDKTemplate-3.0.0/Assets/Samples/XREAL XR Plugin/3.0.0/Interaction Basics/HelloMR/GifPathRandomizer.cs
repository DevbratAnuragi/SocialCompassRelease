using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;

[RequireComponent(typeof(GifSpritePlayer))]
public class GifPathRandomizer : MonoBehaviour
{
    [Header("Randomization Settings")]
    [Tooltip("Add all the possible folder paths from your Resources folder here.")]
    public List<string> resourcePaths = new List<string>();

    private GifSpritePlayer gifPlayer;

    void Awake()
    {
        // --- THE FIX ---
        // Initialize the random number generator with a unique seed based on the current time.
        // This ensures you get a different random sequence every time you run the scene.
        Random.InitState((int)System.DateTime.Now.Ticks);

        gifPlayer = GetComponent<GifSpritePlayer>();

        if (resourcePaths == null || resourcePaths.Count == 0)
        {
            Debug.LogError("[GifPathRandomizer] No resource paths have been assigned in the Inspector.", this);
            return;
        }

        // Now this will produce a different result each time you start.
        int randomIndex = Random.Range(0, resourcePaths.Count);
        string randomPath = resourcePaths[randomIndex];

        Debug.Log($"[GifPathRandomizer] Randomly selected path: '{randomPath}'");

        gifPlayer.resourcesPath = randomPath;
    }

    void Start()
    {
        StartCoroutine(AdjustSizeAfterLoading());
    }

    private IEnumerator AdjustSizeAfterLoading()
    {
        yield return new WaitForEndOfFrame();

        if (gifPlayer.target == null || gifPlayer.frames == null || gifPlayer.frames.Count == 0)
        {
            Debug.LogWarning("[GifPathRandomizer] Could not resize UI. Target Image or frames are not available.", this);
            yield break;
        }

        Sprite firstFrame = gifPlayer.frames[0];
        if (firstFrame == null)
        {
            yield break;
        }

        RectTransform rectTransform = gifPlayer.target.GetComponent<RectTransform>();
        float nativeWidth = firstFrame.rect.width;
        float nativeHeight = firstFrame.rect.height;

        rectTransform.sizeDelta = new Vector2(nativeWidth, nativeHeight);

        Debug.Log($"[GifPathRandomizer] Resized Image to {nativeWidth}x{nativeHeight} to match the new GIF.");
    }
}

