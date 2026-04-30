using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class TapTrialManager : MonoBehaviour
{
    [Header("UI")]
    public TextMeshProUGUI promptText;
    public TextMeshProUGUI statusText;

    [Header("Session Settings")]
    public int trialsPerFinger = 10;
    public float countdownSeconds = 3f;
    public float promptIntervalSeconds = 2f;
    public string sessionPrefix = "session";
    public Image thumbImage;
    public Image indexImage;
    public Image middleImage;
    public Image ringImage;
    public Image pinkyImage;
    

    private string[] fingers = new string[] { "thumb", "index", "middle", "ring", "pinky" };

    private List<TrialRow> rows = new List<TrialRow>();
    private List<string> trialSequence = new List<string>();
    private string sessionId;
    private bool isRunning = false;
    private float sessionStartTime;
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
        public float prompt_time_sec;
        public string notes;

        public TrialRow(int trialId, string finger, float promptTime, string notes = "")
        {
            trial_id = trialId;
            target_finger = finger;
            prompt_time_sec = promptTime;
            this.notes = notes;
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
            case "thumb": thumbImage.color = active; break;
            case "index": indexImage.color = active; break;
            case "middle": middleImage.color = active; break;
            case "ring": ringImage.color = active; break;
            case "pinky": pinkyImage.color = active; break;
        }
    }

        private void ResetFingerColors()
        {
            thumbImage.color = thumbRestColor;
            indexImage.color = indexRestColor;
            middleImage.color = middleRestColor;
            ringImage.color = ringRestColor;
            pinkyImage.color = pinkyRestColor;
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

        ResetFingerColors();
        promptText.text = "SYNC TAP";
        statusText.text = "Tap once strongly now";

        sessionStartTime = Time.time;
        rows.Add(new TrialRow(0, "sync", 0f, "initial sync tap"));

        yield return new WaitForSeconds(1.5f);

        promptText.text = "START";
        ResetFingerColors();
        statusText.text = "";
        yield return new WaitForSeconds(0.5f);

        for (int trial = 1; trial <= trialSequence.Count; trial++)
        {
            string finger = trialSequence[trial - 1];

            float trialStart = Time.time;

            // short pre-cue
            ResetFingerColors();
            promptText.text = "";
            statusText.text = $"Trial {trial} / {trialSequence.Count}";
            yield return new WaitForSeconds(0.15f);

            // actual cue
            HighlightFinger(finger);
            promptText.text = finger.ToUpper();
            statusText.text = $"Trial {trial} / {trialSequence.Count}";

            float promptTime = Time.time - sessionStartTime;
            rows.Add(new TrialRow(trial, finger, promptTime));

            float elapsed = Time.time - trialStart;
            float remaining = Mathf.Max(0f, promptIntervalSeconds - elapsed);
            yield return new WaitForSeconds(remaining);

            ResetFingerColors();
        }

        promptText.text = "DONE";
        statusText.text = "Saving CSV...";
        SaveCsv();

        statusText.text = $"Saved: {sessionId}.csv";
        isRunning = false;
    }

    private void SaveCsv()
    {
        string folder = @"C:/Users/georg/TapDataset/raw_sessions";
        Directory.CreateDirectory(folder);

        string path = Path.Combine(folder, $"{sessionId}_trials.csv");

        using (StreamWriter writer = new StreamWriter(path))
        {
            writer.WriteLine("trial_id,target_finger,prompt_time_sec,notes");

            foreach (var row in rows)
            {
                writer.WriteLine($"{row.trial_id},{row.target_finger},{row.prompt_time_sec:F4},{EscapeCsv(row.notes)}");
            }
        }

        Debug.Log($"Saved CSV to: {path}");
    }

    private string EscapeCsv(string value)
    {
        if (string.IsNullOrEmpty(value))
            return "";

        if (value.Contains(",") || value.Contains("\"") || value.Contains("\n"))
        {
            return $"\"{value.Replace("\"", "\"\"")}\"";
        }

        return value;
    }
}