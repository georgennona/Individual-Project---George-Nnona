using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using TMPro;
using UnityEngine;
using UnityEngine.UI;


public class EvalTapUdpDisplay : MonoBehaviour
{
    [Header("Left Hand")]
    public Image LeftThumb;
    public Image LeftIndex;
    public Image LeftMiddle;
    public Image LeftRing;
    public Image LeftPinky;

    [Header("Right Hand")]
    public Image RightThumb;
    public Image RightIndex;
    public Image RightMiddle;
    public Image RightRing;
    public Image RightPinky;

    [Header("Colors")]
    public Color inactiveColor = new Color(0.5f, 0.5f, 0.5f, 0.45f);
    public Color activeColor   = new Color(0f, 1f, 0f, 0.9f);

    [Header("Logging")]
    public string outputFolder = @"C:/Users/georg/TapDataset/eval_sessions";
    public string sessionPrefix = "eval";

    private UdpClient udpClient;
    private Thread receiveThread;
    private bool running = true;

    private readonly object lockObject = new object();

    private List<string> latestLeft  = new List<string>();
    private List<string> latestRight = new List<string>();

    private float leftTimer  = 0f;
    private float rightTimer = 0f;
    private float displayDuration = 0.2f;

    private StreamWriter csvWriter;
    private string sessionId;

    void Start()
    {
        sessionId = $"{sessionPrefix}_{DateTime.Now:yyyyMMdd_HHmmss}";

        Directory.CreateDirectory(outputFolder);
        string csvPath = Path.Combine(outputFolder, $"{sessionId}_tapxr.csv");
        csvWriter = new StreamWriter(csvPath, append: false);
        csvWriter.WriteLine("timestamp,hand,finger");
        csvWriter.AutoFlush = true;

        Debug.Log($"TapXR logging to: {csvPath}");

        udpClient = new UdpClient(5005);
        receiveThread = new Thread(ReceiveLoop);
        receiveThread.IsBackground = true;
        receiveThread.Start();

        ResetAll();
    }

    void Update()
    {
        leftTimer  -= Time.deltaTime;
        rightTimer -= Time.deltaTime;

        //if (leftTimer  <= 0) latestLeft.Clear();
        //if (rightTimer <= 0) latestRight.Clear();

        //UpdateHandUI(latestLeft,  isLeft: true);
        //UpdateHandUI(latestRight, isLeft: false);
    }

    void UpdateHandUI(List<string> fingers, bool isLeft)
    {
        SetFinger(isLeft, "thumb",  fingers.Contains("thumb"));
        SetFinger(isLeft, "index",  fingers.Contains("index"));
        SetFinger(isLeft, "middle", fingers.Contains("middle"));
        SetFinger(isLeft, "ring",   fingers.Contains("ring"));
        SetFinger(isLeft, "pinky",  fingers.Contains("pinky"));
    }

    void SetFinger(bool isLeft, string finger, bool active)
    {
        Image img = null;

        if (isLeft)
        {
            if (finger == "thumb")  img = LeftThumb;
            if (finger == "index")  img = LeftIndex;
            if (finger == "middle") img = LeftMiddle;
            if (finger == "ring")   img = LeftRing;
            if (finger == "pinky")  img = LeftPinky;
        }
        else
        {
            if (finger == "thumb")  img = RightThumb;
            if (finger == "index")  img = RightIndex;
            if (finger == "middle") img = RightMiddle;
            if (finger == "ring")   img = RightRing;
            if (finger == "pinky")  img = RightPinky;
        }

        if (img != null)
            img.color = active ? activeColor : inactiveColor;
    }

    void ResetAll()
    {
        UpdateHandUI(new List<string>(), isLeft: true);
        UpdateHandUI(new List<string>(), isLeft: false);
    }

    void ReceiveLoop()
    {
        IPEndPoint endPoint = new IPEndPoint(IPAddress.Any, 5005);

        while (running)
        {
            try
            {
                byte[] data = udpClient.Receive(ref endPoint);
                string json  = Encoding.UTF8.GetString(data);

                TapMessage msg = JsonUtility.FromJson<TapMessage>(json);
                if (msg == null || msg.fingers == null || msg.fingers.Length == 0)
                    continue;

                // only log right hand
                if (msg.hand == "right" && msg.fingers.Length == 1)
                {
                    string finger = msg.fingers[0];
                    double ts     = msg.timestamp;

                    lock (lockObject)
                    {
                        csvWriter?.WriteLine($"{ts:F6},{msg.hand},{finger}");
                    }
                }

                lock (lockObject)
                {
                    if (msg.hand == "left")
                    {
                        latestLeft = new List<string>(msg.fingers);
                        leftTimer  = displayDuration;
                    }
                    else if (msg.hand == "right")
                    {
                        latestRight = new List<string>(msg.fingers);
                        rightTimer  = displayDuration;
                    }
                }
            }
            catch (Exception e)
            {
                if (running)
                    Debug.LogWarning("UDP receive error: " + e.Message);
            }
        }
    }

    void OnApplicationQuit()
    {
        running = false;

        try { udpClient?.Close();       } catch { }
        try { receiveThread?.Interrupt(); } catch { }
        try { csvWriter?.Close();        } catch { }
    }
}
