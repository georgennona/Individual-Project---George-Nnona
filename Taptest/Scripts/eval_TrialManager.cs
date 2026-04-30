using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class EvalTrialManager : MonoBehaviour
{
    [Header("UI")]
    public TextMeshProUGUI promptText;
    public TextMeshProUGUI statusText;

    [Header("Session Settings")]
    public int trialsPerFinger = 20;
    public float countdownSeconds = 3f;
    public float promptIntervalSeconds = 2f;
    public string sessionPrefix = "eval";
    public Image thumbImage;
    public Image indexImage;
    public Image middleImage;
    public Image ringImage;
    public Image pinkyImage;

    [Header("Output")]
    public string outputFolder = @"C:/Users/georg/TapDataset/eval_sessions";

    private string[] fingers = new string[] { "thumb", "index", "middle", "ring", "pinky" };

    private List<TrialRow> rows = new List<TrialRow>();
    private List<string> trialSequence = new List<string>();
    private string sessionId;
    private bool isRunning = false;

    private Color thumbRestColor;
    private Color indexRestColor;
    private Color middleRestColor;
    private Color ringRestColor;
    private Color pinkyRestColor;

    [Serializable]
    public class TrialRow
    {
        public int trial_id;
        public string target_finger;
        public double prompt_unix_time;

        public TrialRow(int trialId, string finger, double unixTime)
        {
            trial_id = trialId;
            target_finger = finger;
            prompt_unix_time = unixTime;
        }
    }

    private void Start()
    {
        thumbRestColor = thumbImage.color;
        indexRestColor = indexImage.color;
        middleRestColor = middleImage.color;
        ringRestColor = ringImage.color;
        pinkyRestColor = pinkyImage.color;

        ResetFingerColors();
        sessionId = $"{sessionPrefix}_{DateTime.Now:yyyyMMdd_HHmmss}";
        promptText.text = "Press Space to Start";
        statusText.text = "Ready";
    }

    private void Update()
    {
        if (!isRunning && Input.GetKeyDown(KeyCode.Space))
        {
            StartCoroutine(RunSession());
        }
    }

    private double GetUnixTime()
    {
        return (DateTime.UtcNow - new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc)).TotalSeconds;
    }

    private void Shuffle(List<string> list)
    {
        for (int i = 0; i < list.Count; i++)
        {
            int j = UnityEngine.Random.Range(i, list.Count);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }

    private void BuildBalancedTrialSequence(int trialsPerFinger)
    {
        trialSequence.Clear();

        foreach (string finger in fingers)
        {
            for (int i = 0; i < trialsPerFinger; i++)
            {
                trialSequence.Add(finger);
            }
        }

        Shuffle(trialSequence);
    }

    private void HighlightFinger(string finger)
    {
        ResetFingerColors();
        Color active = Color.yellow;

        switch (finger)
        {
            case "thumb":  thumbImage.color  = active; break;
            case "index":  indexImage.color  = active; break;
            case "middle": middleImage.color = active; break;
            case "ring":   ringImage.color   = active; break;
            case "pinky":  pinkyImage.color  = active; break;
        }
    }

    private void ResetFingerColors()
    {
        thumbImage.color  = thumbRestColor;
        indexImage.color  = indexRestColor;
        middleImage.color = middleRestColor;
        ringImage.color   = ringRestColor;
        pinkyImage.color  = pinkyRestColor;
    }

    private IEnumerator RunSession()
    {
        isRunning = true;
        rows.Clear();
        sessionId = $"{sessionPrefix}_{DateTime.Now:yyyyMMdd_HHmmss}";

        BuildBalancedTrialSequence(trialsPerFinger);
        ResetFingerColors();

        for (int i = Mathf.CeilToInt(countdownSeconds); i > 0; i--)
        {
            promptText.text = i.ToString();
            statusText.text = "Get ready";
            yield return new WaitForSeconds(1f);
        }

        promptText.text = "GO";
        statusText.text = "";
        yield return new WaitForSeconds(0.5f);

        for (int trial = 1; trial <= trialSequence.Count; trial++)
        {
            string finger = trialSequence[trial - 1];
            float trialStart = Time.time;

            ResetFingerColors();
            promptText.text = "";
            statusText.text = $"Trial {trial} / {trialSequence.Count}";
            yield return new WaitForSeconds(0.15f);

            // Record unix time at prompt
            double promptUnixTime = GetUnixTime();

            HighlightFinger(finger);
            promptText.text = finger.ToUpper();
            statusText.text = $"Trial {trial} / {trialSequence.Count}";

            rows.Add(new TrialRow(trial, finger, promptUnixTime));

            float elapsed = Time.time - trialStart;
            float remaining = Mathf.Max(0f, promptIntervalSeconds - elapsed);
            yield return new WaitForSeconds(remaining);

            ResetFingerColors();
        }

        promptText.text = "DONE";
        statusText.text = "Saving...";
        SaveCsv();

        statusText.text = $"Saved: {sessionId}_trials.csv";
        isRunning = false;
    }

    private void SaveCsv()
    {
        Directory.CreateDirectory(outputFolder);
        string path = Path.Combine(outputFolder, $"{sessionId}_trials.csv");

        using (StreamWriter writer = new StreamWriter(path))
        {
            writer.WriteLine("trial_id,target_finger,prompt_unix_time");

            foreach (var row in rows)
            {
                writer.WriteLine($"{row.trial_id},{row.target_finger},{row.prompt_unix_time:F6}");
            }
        }

        Debug.Log($"Saved eval trials to: {path}");
    }
}
