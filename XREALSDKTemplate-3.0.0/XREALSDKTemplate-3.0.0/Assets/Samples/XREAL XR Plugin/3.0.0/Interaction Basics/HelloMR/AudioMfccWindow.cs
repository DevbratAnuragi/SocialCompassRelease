// AudioMfccWindow.cs
using System;
using System.Collections;
using System.Linq;
using UnityEngine;

public class AudioMfccWindow : MonoBehaviour
{
    [Header("Audio")]
    public int sampleRate = 16000;
    public int windowSeconds = 20;
    public int strideSeconds = 20;

    [Header("MFCC")]
    public int melBands = 40;
    public int mfccDims = 13;
    public int winMs = 25;
    public int hopMs = 10;
    public bool doPreEmphasis = true;

    public event Action<float[], double> OnMfccWindowReady;

    AudioClip _clip;
    float[] _ring;
    int _ringWrite;
    int _ringSamples;

    void OnEnable()
    {
        _ringSamples = sampleRate * windowSeconds * 2;
        _ring = new float[_ringSamples];
        StartCoroutine(Capture());
    }

    void OnDisable()
    {
        if (_clip != null && Microphone.IsRecording(null)) Microphone.End(null);
    }

    IEnumerator Capture()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        while (!UnityEngine.Android.Permission.HasUserAuthorizedPermission(UnityEngine.Android.Permission.Microphone))
        {
            UnityEngine.Android.Permission.RequestUserPermission(UnityEngine.Android.Permission.Microphone);
            yield return new WaitForSeconds(1.0f);
        }
#endif

        int need = sampleRate * windowSeconds;
        int stride = sampleRate * strideSeconds;
        long samplesWritten = 0;
        long nextEmitStart = -1;
        
        _clip = Microphone.Start(null, true, windowSeconds * 2, sampleRate);
        while (Microphone.GetPosition(null) <= 0) yield return null;
        
        int pos0 = Microphone.GetPosition(null);
        // Use the shared monotonic clock
        double monotonicTimeAtMicStart = MonoClock.Now() - (double)pos0 / sampleRate;
        samplesWritten = pos0;
        
        nextEmitStart = Math.Max(0, samplesWritten - need);
        nextEmitStart -= nextEmitStart % stride;
        
        int lastPos = pos0;

        while (true)
        {
            int pos = Microphone.GetPosition(null);

            if (pos < lastPos) // Wrapped
            {
                int tail = _clip.samples - lastPos;
                int head = pos;

                if (tail > 0 || head > 0)
                {
                    var tmp = new float[tail + head];
                    if (tail > 0) _clip.GetData(tmp, lastPos);
                    if (head > 0)
                    {
                        var headBuf = new float[head];
                        _clip.GetData(headBuf, 0);
                        Array.Copy(headBuf, 0, tmp, tail, head);
                    }
                    for (int i = 0; i < tmp.Length; i++)
                    {
                        _ring[_ringWrite] = tmp[i];
                        _ringWrite = (_ringWrite + 1) % _ringSamples;
                    }
                    samplesWritten += tmp.Length;
                }
            }
            else // No wrap
            {
                int toGet = pos - lastPos;
                if (toGet > 0)
                {
                    var tmp = new float[toGet];
                    _clip.GetData(tmp, lastPos);
                    for (int i = 0; i < toGet; i++)
                    {
                        _ring[_ringWrite] = tmp[i];
                        _ringWrite = (_ringWrite + 1) % _ringSamples;
                    }
                    samplesWritten += toGet;
                }
            }
            lastPos = pos;
            
            while (nextEmitStart >= 0 && samplesWritten - nextEmitStart >= need)
            {
                int rel = (int)(samplesWritten - nextEmitStart);
                int readStart = (_ringWrite - rel + _ringSamples) % _ringSamples;
                var audio = new float[need];
                for (int i = 0; i < need; i++) audio[i] = _ring[(readStart + i) % _ringSamples];

                var mfcc = Mfcc13(audio, sampleRate, winMs, hopMs, melBands, mfccDims, doPreEmphasis);

                if (mfcc.Length > 0)
                {
                    var mfcc430 = ResampleTime(mfcc, 430);
                    CmvnInPlace(mfcc430);
                    var flat = Flatten(mfcc430);
                    double timestamp = monotonicTimeAtMicStart + (nextEmitStart) / sampleRate;

                    Debug.Log($"[AUDIO] window center {timestamp:F3}");
                    OnMfccWindowReady?.Invoke(flat, timestamp);
                }
                nextEmitStart += stride;
            }
            yield return null;
        }
    }

    // ======== MFCC implementation (minimal, CPU-friendly) ========

    static float[][] Mfcc13(float[] mono, int sr, int winMs, int hopMs, int mels, int dims, bool pre)
    {
        if (pre) PreEmphasisInPlace(mono, 0.97f);
        int win = Mathf.RoundToInt(sr * winMs / 1000f);
        int hop = Mathf.RoundToInt(sr * hopMs / 1000f);
        int fftN = NextPow2(win);
        var ham = Hamming(win);

        int frames = 1 + (mono.Length - win) / hop;
        if (frames < 1) return new float[0][];
        var melFb = MelFilterbank(sr, fftN / 2 + 1, mels);
        var dct = DctMatrix(mels, dims);

        var outFeat = new float[frames][];
        var re = new double[fftN];
        var im = new double[fftN];

        for (int f = 0; f < frames; f++)
        {
            int off = f * hop;
            Array.Clear(re, 0, fftN); Array.Clear(im, 0, fftN);
            for (int i = 0; i < win; i++)
            {
                double s = mono[off + i] * ham[i];
                re[i] = s;
            }
            FftInPlace(re, im);

            var pow = new double[fftN / 2 + 1];
            for (int k = 0; k < pow.Length; k++)
            {
                double mag = re[k] * re[k] + im[k] * im[k];
                pow[k] = mag;
            }

            var mel = new double[mels];
            for (int m = 0; m < mels; m++)
            {
                double sum = 0;
                var fb = melFb[m];
                for (int k = 0; k < fb.Length; k++) sum += pow[k] * fb[k];
                mel[m] = Math.Log(sum + 1e-10);
            }

            var mf = new float[dims];
            for (int i = 0; i < dims; i++)
            {
                double v = 0;
                for (int j = 0; j < mels; j++) v += dct[i][j] * mel[j];
                mf[i] = (float)v;
            }
            outFeat[f] = mf;
        }
        return outFeat;
    }

    static void PreEmphasisInPlace(float[] x, float a)
    { for (int i = x.Length - 1; i >= 1; i--) x[i] = x[i] - a * x[i - 1]; }

    static int NextPow2(int n) { int p = 1; while (p < n) p <<= 1; return p; }

    static float[] Hamming(int n)
    {
        var w = new float[n];
        for (int i = 0; i < n; i++) w[i] = 0.54f - 0.46f * Mathf.Cos(2 * Mathf.PI * i / (n - 1));
        return w;
    }

    static double[][] MelFilterbank(int sr, int bins, int mels)
    {
        double HzToMel(double hz) => 2595.0 * Math.Log10(1 + hz / 700.0);
        double MelToHz(double mel) => 700.0 * (Math.Pow(10, mel / 2595.0) - 1);
        double fMin = 0, fMax = sr / 2.0;
        var melMin = HzToMel(fMin); var melMax = HzToMel(fMax);
        var melPts = new double[mels + 2];
        for (int i = 0; i < melPts.Length; i++) melPts[i] = melMin + (melMax - melMin) * i / (mels + 1);
        var hz = new double[mels + 2]; for (int i = 0; i < hz.Length; i++) hz[i] = MelToHz(melPts[i]);
        var bin = new int[mels + 2]; for (int i = 0; i < bin.Length; i++) bin[i] = (int)Math.Floor((bins - 1) * hz[i] / (sr / 2.0));

        var fb = new double[mels][];
        for (int m = 1; m <= mels; m++)
        {
            var v = new double[bins];
            for (int k = bin[m - 1]; k < bin[m]; k++) v[k] = (k - bin[m - 1]) / (double)(bin[m] - bin[m - 1] + 1e-9);
            for (int k = bin[m]; k < bin[m + 1]; k++) v[k] = (bin[m + 1] - k) / (double)(bin[m + 1] - bin[m] + 1e-9);
            fb[m - 1] = v;
        }
        return fb;
    }

    static double[][] DctMatrix(int nIn, int nOut)
    {
        var m = new double[nOut][];
        double norm0 = Math.Sqrt(1.0 / nIn);
        double norm = Math.Sqrt(2.0 / nIn);
        for (int i = 0; i < nOut; i++)
        {
            m[i] = new double[nIn];
            for (int j = 0; j < nIn; j++)
            {
                double c = Math.Cos(Math.PI * (j + 0.5) * i / nIn);
                m[i][j] = (i == 0 ? norm0 : norm) * c;
            }
        }
        return m;
    }

    static float[][] ResampleTime(float[][] feat, int targetFrames)
    {
        int F = feat.Length; if (F == 0 || F == targetFrames) return feat;
        int D = feat[0].Length; var outF = new float[targetFrames][];
        for (int t = 0; t < targetFrames; t++)
        {
            double pos = (double)t * (F - 1) / (targetFrames - 1);
            int i = (int)Math.Floor(pos); int j = Math.Min(i + 1, F - 1); double a = pos - i;
            var row = new float[D];
            for (int d = 0; d < D; d++) row[d] = (float)((1 - a) * feat[i][d] + a * feat[j][d]);
            outF[t] = row;
        }
        return outF;
    }

    static void CmvnInPlace(float[][] x)
    {
        if (x.Length == 0) return;
        int T = x.Length, D = x[0].Length;
        var mean = new double[D]; var var_ = new double[D];
        for (int t = 0; t < T; t++) for (int d = 0; d < D; d++) mean[d] += x[t][d];
        for (int d = 0; d < D; d++) mean[d] /= T;
        for (int t = 0; t < T; t++) for (int d = 0; d < D; d++) { double z = x[t][d] - mean[d]; var_[d] += z * z; }
        for (int d = 0; d < D; d++) var_[d] = Math.Sqrt(var_[d] / Math.Max(1, T) + 1e-9); // Use T not T-1 for consistency
        for (int t = 0; t < T; t++) for (int d = 0; d < D; d++) x[t][d] = (float)((x[t][d] - mean[d]) / var_[d]);
    }

    static float[] Flatten(float[][] x)
    {
        if (x.Length == 0) return new float[0];
        int T = x.Length, D = x[0].Length; var y = new float[T * D]; int k = 0;
        for (int t = 0; t < T; t++) for (int d = 0; d < D; d++) y[k++] = x[t][d];
        return y;
    }

    static void FftInPlace(double[] re, double[] im)
    {
        int n = re.Length;
        for (int i = 1, j = 0; i < n; i++)
        {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j &= ~bit;
            j |= bit;
            if (i < j) { var tr = re[i]; re[i] = re[j]; re[j] = tr; tr = im[i]; im[i] = im[j]; im[j] = tr; }
        }
        for (int len = 2; len <= n; len <<= 1)
        {
            double ang = -2 * Math.PI / len;
            double wlenRe = Math.Cos(ang), wlenIm = Math.Sin(ang);
            for (int i = 0; i < n; i += len)
            {
                double wr = 1, wi = 0;
                for (int j = 0; j < len / 2; j++)
                {
                    int u = i + j, v = i + j + len / 2;
                    double xr = re[v] * wr - im[v] * wi;
                    double xi = re[v] * wi + im[v] * wr;
                    re[v] = re[u] - xr; im[v] = im[u] - xi;
                    re[u] += xr; im[u] += xi;
                    double wrn = wr * wlenRe - wi * wlenIm;
                    wi = wr * wlenIm + wi * wlenRe; wr = wrn;
                }
            }
        }
    }

    static double Now() => DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0;
}