Of course\! Here is the guide converted into a clean and readable Markdown format.

# End-to-End Guide: Local LLM Server & Unity/XREAL App

A quick guide to run the local LLM server and dashboard, and build/run the Unity app on XREAL/Android.

### 0\) What’s in this repo

```
.
├─ LLMserver/
│  ├─ server.py            # WebSocket backend (ingests IMU/MFCC windows, returns prompts)
│  ├─ stream_app.py        # Streamlit dashboard for live telemetry & prompts
│  └─ requirements.txt     # Python deps
│
├─ SocialCompassRelease/
│  └─ XREALSDKTemplate-3.0.0/
│     └─ XREALSDKTemplate-3.0.0/   # Unity project root
│        └─ Assets/…               # Use scene: HelloMR
│
└─ Release/
   └─ SocialCompass.apk    # Prebuilt Android APK (optional)
```

### 1\) Prerequisites

  * **Python 3.9–3.11**
  * `pip` (or an alternative like `uv`/`poetry`)
  * **Streamlit** (installed via requirements)
  * **(Optional, recommended)** [**Ollama**](https://ollama.com/) with a small instruct model if `server.py` is configured to call an LLM locally.
  * **Unity 2021/2022 LTS** (the project uses XREAL SDK 3.0.0—match the Unity version you use for this project).
  * **Android SDK/NDK, ADB**, and XREAL/NRSDK dependencies set up as usual.
  * A phone/Beam Pro device and your dev machine on the **same Wi-Fi/LAN**.

### 2\) Install Python dependencies

1.  Navigate to the `LLMserver` directory.
2.  Create and activate a virtual environment:
    ```bash
    cd LLMserver
    python -m venv .venv

    # On Windows:
    .venv\Scripts\activate

    # On macOS/Linux:
    source .venv/bin/activate
    ```
3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
4.  If using Ollama, ensure it’s running and your model is pulled (e.g., run `ollama run llama3.1:8b-instruct` once).

### 3\) Configure networking

The WebSocket URL in the Unity app is set in `Assets/Scripts/HaseNetClient.cs`. Change the IP to your development machine’s LAN IP address.

```csharp
public string serverWs = "ws://192.168.1.100:8765";
```

### 4\) Run the backend + dashboard

Open two separate terminals in the `LLMserver/` directory.

  * **Terminal 1: Backend (WebSocket server)**

    ```bash
    python server.py
    ```

  * **Terminal 2: Dashboard (Streamlit)**

    ```bash
    streamlit run stream_app.py
    ```

The server will listen on `ws://<your-ip>:8765`. The Streamlit dashboard will open in your browser, typically at `http://localhost:8501`.

### 5\) Unity project: open & build

1.  Open the Unity project located at:
    `SocialCompassRelease/XREALSDKTemplate-3.0.0/XREALSDKTemplate-3.0.0/`

2.  Open the **HelloMR** scene:
    `Assets/Scenes/HelloMR.unity`

3.  In the scene hierarchy, locate the object that has the `HaseNetClient` component and configure it:

      * Set **Server Ws** to `ws://<your-dev-machine-ip>:8765`.
      * Optionally enable `receiveServerPrompts` if you want on-device prompt toasts.
      * Ensure these components are wired correctly in the Inspector:
          * `WindowMux` has references to `BeamImu40Hz` and `AudioMfccWindow`.
          * `HaseNetClient.mux` is set to the `WindowMux` instance.
          * (Optional) `HaseNetClient.prompt` is set to the `PromptSlide` UI.

4.  **Android Build Settings:**

      * **Platform:** Android
      * **Architecture:** ARM64
      * **Scripting Backend:** IL2CPP
      * **Minimum API Level:** As required by your NRSDK/XREAL SDK version.
      * **Player Settings → Permissions:** Ensure **Microphone** is requested.
      * Ensure orientation & XR settings are configured per XREAL SDK guidance.

5.  **Build & Run** to your Beam Pro or phone.

### 6\) Using the prebuilt APK

If you prefer not to build from source:

1.  Make sure your LLM server is running.
2.  Install the APK via ADB:
    ```bash
    adb install -r Release/SocialCompass.apk
    ```
    > **Note:** If the prebuilt APK has a different hardcoded server IP, you must either rebuild the app or use a local proxy/port-forwarding workaround.

### 7\) Data & model expectations

  * **IMU:** 6 channels `[ax, ay, az, gx, gy, gz]`, sampled at 40 Hz in 20-second windows (→ `800×6` shape). Units are m/s² and rad/s.
  * **Audio/MFCC:** 13-dimensional MFCC, resampled to 430 frames per 20-second window, with per-window CMVN normalization (→ `430×13` shape).

### 8\) Troubleshooting

  * **No Prompts / Socket Stuck:**
      * Verify your laptop's IP and check firewall settings. Can the device ping your machine?
      * Confirm `server.py` is running and prints connection logs.
      * You should see `[HASE-NET] Connected ws://…` in the Unity Console.
  * **IMU Zeros / Gyro Not Changing:**
      * On Android, ensure Motion sensor permissions are granted.
      * If using the Unity fallback, ensure `Input.gyro.enabled = true` is called.
      * Move the device physically; the gyro will be near zero if the device is perfectly still.
  * **Audio MFCC Not Firing:**
      * Microphone permission must be granted at runtime.
      * Check logs from `AudioMfccWindow`; it should print when windows are emitted.
  * **Prompt UI Appears on Scene Start:**
      * You are likely calling `PromptSlide.ShowOnce()` too early. Use a flag to only show prompts after the first real data window has been sent.
  * **Dashboard is Empty:**
      * Check for `[HASE-NET] OnWindow imu=… mfcc=…` in Unity logs to confirm data is being sent.
      * Check the Streamlit terminal for any exceptions.

### 9\) Useful commands

  * **Find your machine’s IP (macOS/Linux):**
    ```bash
    ip addr
    # or
    ifconfig
    ```
  * **View ADB logs from the device:**
    ```bash
    adb logcat | grep -i socialcompass
    ```
  * **Uninstall the app:**
    ```bash
    adb uninstall com.Demo.SocialCompass
    ```

### 10\) Notes & customization

  * Change the LLM model and parameters in `server.py` to fit your latency and tone requirements.
  * If you deploy the server to another host, update the `serverWs` URL in Unity accordingly.
  * To adjust prompt behavior (e.g., debouncing), see `HaseNetClient.HandleServerJson`.
