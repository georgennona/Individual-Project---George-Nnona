using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

[Serializable]
public class TapMessage
{
    public string hand;
    public string[] fingers;
    public double timestamp;
}

public class TapUdpDisplay : MonoBehaviour
{
    // LEFT HAND
    public Image LeftThumb;
    public Image LeftIndex;
    public Image LeftMiddle;
    public Image LeftRing;
    public Image LeftPinky;

    // RIGHT HAND
    public Image RightThumb;
    public Image RightIndex;
    public Image RightMiddle;
    public Image RightRing;
    public Image RightPinky;

    public Color inactiveColor = new Color(0.5f, 0.5f, 0.5f, 0.45f);
    public Color activeColor = new Color(0f, 1f, 0f, 0.9f);

    private UdpClient udpClient;
    private Thread receiveThread;
    private bool running = true;

    private readonly object lockObject = new object();

    private List<string> latestLeft = new List<string>();
    private List<string> latestRight = new List<string>();

    private float leftTimer = 0f;
    private float rightTimer = 0f;

    private float displayDuration = 0.2f;

    void Start()
    {
        udpClient = new UdpClient(5005);
        receiveThread = new Thread(ReceiveLoop);
        receiveThread.IsBackground = true;
        receiveThread.Start();

        ResetAll();
    }

    void Update()
    {
        leftTimer -= Time.deltaTime;
        rightTimer -= Time.deltaTime;

        if (leftTimer <= 0)
        {
            latestLeft.Clear();
        }

        if (rightTimer <= 0)
        {
            latestRight.Clear();
        }

        UpdateHandUI(latestLeft, true);
        UpdateHandUI(latestRight, false);
    }

    void UpdateHandUI(List<string> fingers, bool isLeft)
    {
        SetFinger(isLeft, "thumb", fingers.Contains("thumb"));
        SetFinger(isLeft, "index", fingers.Contains("index"));
        SetFinger(isLeft, "middle", fingers.Contains("middle"));
        SetFinger(isLeft, "ring", fingers.Contains("ring"));
        SetFinger(isLeft, "pinky", fingers.Contains("pinky"));
    }

    void SetFinger(bool isLeft, string finger, bool active)
    {
        Image img = null;

        if (isLeft)
        {
            if (finger == "thumb") img = LeftThumb;
            if (finger == "index") img = LeftIndex;
            if (finger == "middle") img = LeftMiddle;
            if (finger == "ring") img = LeftRing;
            if (finger == "pinky") img = LeftPinky;
        }
        else
        {
            if (finger == "thumb") img = RightThumb;
            if (finger == "index") img = RightIndex;
            if (finger == "middle") img = RightMiddle;
            if (finger == "ring") img = RightRing;
            if (finger == "pinky") img = RightPinky;
        }

        if (img != null)
        {
            img.color = active ? activeColor : inactiveColor;
        }
    }

    void ResetAll()
    {
        UpdateHandUI(new List<string>(), true);
        UpdateHandUI(new List<string>(), false);
    }

    void ReceiveLoop()
    {
        IPEndPoint endPoint = new IPEndPoint(IPAddress.Any, 5005);

        while (running)
        {
            try
            {
                byte[] data = udpClient.Receive(ref endPoint);
                string json = Encoding.UTF8.GetString(data);

                TapMessage msg = JsonUtility.FromJson<TapMessage>(json);
                if (msg == null || msg.fingers == null) continue;

                lock (lockObject)
                {
                    if (msg.hand == "left")
                    {
                        latestLeft = new List<string>(msg.fingers);
                        leftTimer = displayDuration;
                    }
                    else if (msg.hand == "right")
                    {
                        latestRight = new List<string>(msg.fingers);
                        rightTimer = displayDuration;
                    }
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning("UDP receive error: " + e.Message);
            }
        }
    }

    void OnApplicationQuit()
    {
        running = false;

        try { udpClient?.Close(); } catch { }
        try { receiveThread?.Interrupt(); } catch { }
    }
}