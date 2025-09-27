using UnityEngine;
using UnityEngine.Android; // Required for Android permissions

public class PermissionManager : MonoBehaviour
{
    void Start()
    {
        // Check if we already have microphone permission
        if (!Permission.HasUserAuthorizedPermission(Permission.Microphone))
        {
            // If not, request it from the user
            Permission.RequestUserPermission(Permission.Microphone);
        }
    }
}