#include <ns3/constant-position-mobility-model.h>
#include <ns3/core-module.h>
#include <ns3/log.h>
#include <ns3/lr-wpan-module.h>
#include <ns3/packet.h>
#include <ns3/propagation-delay-model.h>
#include <ns3/propagation-loss-model.h>
#include <ns3/simulator.h>
#include "ns3/network-module.h"
#include <ns3/single-model-spectrum-channel.h>
#include <iostream>
#include <deque>
#include <cstdlib>
#include "ns3/energy-module.h"


using namespace ns3;
double totalEnergy = 0;
double totalDelay;
double unitEnergy = 30;
double txPower = 57.42;
double sleepPower = 1.4;
double unitDelay = 4.0;
ns3::Time lastPacketReceivedTime;
ns3::Time firstPacketSentTime ;
ns3::Time latencyAccumulator ;
uint8_t MacMaxFrameRetries = 7;
double emaChannelState = 1.0; // Global variable
const double alpha = 0.1; // Smoothing factor, adjust as needed
int numSentPackets =0;
int numReceivedPackets = 0;

static std::map<uint8_t, int> retransmissionCounts;
double baseSuperframeDuration = 0.01536; // 15.36 milliseconds
int bcnOrd = 10; // Example beacon order
double beaconInterval = std::pow(2, bcnOrd) * baseSuperframeDuration;
std::deque<int> transmissionHistory;
const size_t historySize = 5; // Size for tracking the last three packets
// Constants (tweak these based on your scenario)
const double transmissionPower = 45;  // Power can be adjusted
uint64_t numSentPacketsLastBeacon = 0;
uint64_t numFailedPacketsLastBeacon = 0;
double lastFailureRate = 0.0; // Failure rate from the last beacon interval
static double g_totalTxCost = 0.0;
static double g_totalTxbackoff = 0.0;
double sleepEnergy = 0;
double txEnergy = 0;
double energyDiff ;
double txTime;

Ptr<LrWpanNetDevice> g_device;
Ptr<LrWpanNetDevice> g_coordinatorDevice;

// Implement the BackoffTimeHandler function
    void BackoffTimeHandler(double backoffTime) {
    NS_LOG_UNCOND("Current Backoff Time: " << backoffTime << " secs");
    // Additional processing or accumulation of backoff times
    g_totalTxbackoff += backoffTime ;
    sleepEnergy = txPower*backoffTime;

}

void TransactionTimeHandler(uint32_t transactionSymbols) {
    double symbolRate = 62500; // Example: IEEE 802.15.4 standard symbol rate
    txTime = (double)transactionSymbols / symbolRate;

    // Accumulate the transaction cost
    g_totalTxCost += txTime;
    txEnergy = txTime*txPower;

    NS_LOG_UNCOND("Current Transaction Time: " << txTime << " secs, Total Transaction Cost: " << g_totalTxCost << " secs");
}

static void SendPacket(Ptr<LrWpanNetDevice> device)
{

        Ptr<Packet> p = Create<Packet>(100);
        McpsDataRequestParams params;
        params.m_dstPanId = 5;
        params.m_srcAddrMode = SHORT_ADDR;
        params.m_dstAddrMode = SHORT_ADDR;
        params.m_dstAddr = Mac16Address("00:01");
        params.m_msduHandle = 0; // or some unique handle
        params.m_txOptions = TX_OPTION_ACK;
        device->GetMac()->McpsDataRequest(params, p);
        numSentPackets++;
        numSentPacketsLastBeacon++;

        //Simulator::Schedule(Seconds(0.01), &SendPacket, device);

}

static void BeaconIndication (MlmeBeaconNotifyIndicationParams params, Ptr<Packet> p)
{
  NS_LOG_UNCOND (Simulator::Now ().GetSeconds () << " secs | Received BEACON packet of size " << p->GetSize ());
}

static void DataIndication (McpsDataIndicationParams params, Ptr<Packet> p)
{
  NS_LOG_UNCOND (Simulator::Now ().GetSeconds () << " secs | Received DATA packet of size " << p->GetSize ());
}

static void TransEndIndication (McpsDataConfirmParams params)
{

        while (transmissionHistory.size() > historySize)
    {
        transmissionHistory.pop_front();
    }
    if (params.m_status == LrWpanMcpsDataConfirmStatus::IEEE_802_15_4_SUCCESS)
    {

        Ptr<LrWpanCsmaCa> csmaCa = g_device->GetMac()->GetCsmaCa();
        uint8_t defaultMacMinBE = 3;
        numReceivedPackets++;
        int retries = g_device->GetMac()->GetRetransmissionCount();
        double energyForThisAttempt = unitEnergy * (retries + 1);
        totalEnergy += energyForThisAttempt;
        double Delay = unitDelay * (retries + 1);
        csmaCa->SetMacMinBE(defaultMacMinBE);
        NS_LOG_UNCOND("MacMinBE reset to default: " << (uint32_t)defaultMacMinBE);
        transmissionHistory.push_back(1);
        totalDelay += Delay;
        energyDiff = txEnergy - sleepEnergy;
        NS_LOG_UNCOND (Simulator::Now ().GetSeconds () << " secs | Transmission successfully sent after "
                      << retries << " retries. EMA: " << emaChannelState);


        NS_LOG_UNCOND (Simulator::Now ().GetSeconds () << " secs | energy for this attempt "
                      << energyDiff );
        Simulator::Schedule(Seconds(0.3), &SendPacket, g_device);

    }

   else if (params.m_status == LrWpanMcpsDataConfirmStatus::IEEE_802_15_4_NO_ACK)
    {


        numFailedPacketsLastBeacon++;
        double energyForThisAttempt = unitEnergy * MacMaxFrameRetries;
        totalEnergy += energyForThisAttempt;
        double Delay = unitDelay * MacMaxFrameRetries;
        transmissionHistory.push_back(0);
        totalDelay += Delay;
        energyDiff = txEnergy - sleepEnergy;
        int retries = g_device->GetMac()->GetRetransmissionCount();
        NS_LOG_UNCOND (Simulator::Now ().GetSeconds () << " secs | Packet failed after "
                      << retries << " retries. ");
        NS_LOG_UNCOND (Simulator::Now ().GetSeconds () << " secs | energy for this attempt "
                      << energyDiff );

        Simulator::Schedule(Seconds(0.3), &SendPacket, g_device);

    }

    int successfulTransmissions = std::count(transmissionHistory.begin(), transmissionHistory.end(), 1);
    double currentValue = static_cast<double>(successfulTransmissions) / transmissionHistory.size();
    emaChannelState = (1 - alpha) * currentValue + alpha * emaChannelState;
     NS_LOG_UNCOND (Simulator::Now ().GetSeconds () << "  EMA: " << emaChannelState);

}


static void DataIndicationCoordinator (McpsDataIndicationParams params, Ptr<Packet> p)
{
  NS_LOG_UNCOND (Simulator::Now ().GetSeconds () << "s Coordinator Received DATA packet (size " << p->GetSize () << " bytes)");

}

static void StartConfirm (MlmeStartConfirmParams params)
{
  if (params.m_status == MLMESTART_SUCCESS)
    {
      NS_LOG_UNCOND (Simulator::Now ().GetSeconds () << "Beacon status SUCESSFUL");
    }
}

void EndOfBeaconInterval() {
    // Calculate the failure rate
    double failureRate = (numFailedPacketsLastBeacon * 100.0) / numSentPacketsLastBeacon;
    lastFailureRate = failureRate; // Store for use in the next interval

    // Log the failure rate
    NS_LOG_UNCOND("Failure Rate at the end of the Beacon Interval: " << failureRate << "%");

    // Reset counters for the next interval
    numSentPacketsLastBeacon = 0;
    numFailedPacketsLastBeacon = 0;

    // Schedule the next calculation
    Simulator::Schedule(Seconds(beaconInterval), &EndOfBeaconInterval);
}

void OnRetransmission(uint32_t retransmissionCount) {
    NS_LOG_UNCOND(Simulator::Now ().GetSeconds ()<<" Retransmission attempt number: " << retransmissionCount);

    numFailedPacketsLastBeacon++;
    numSentPacketsLastBeacon++;
     transmissionHistory.push_back(0);
     NS_LOG_UNCOND (Simulator::Now ().GetSeconds () << " secs | transaction time is "
                      << txTime);
    energyDiff = txEnergy - sleepEnergy;

     NS_LOG_UNCOND (Simulator::Now ().GetSeconds () << " secs | energy for this attempt "
                      << energyDiff );

}



int main (int argc, char *argv[])
{

  LogComponentEnableAll (LOG_PREFIX_TIME);
  LogComponentEnableAll (LOG_PREFIX_FUNC);
  //LogComponentEnable ("LrWpanMac", LOG_LEVEL_INFO);
  //LogComponentEnable ("LrWpanCsmaCa", LOG_LEVEL_INFO);


  LrWpanHelper lrWpanHelper;

  // Create 2 nodes, and a NetDevice for each one
  Ptr<Node> n0 = CreateObject <Node> ();
  Ptr<Node> n1 = CreateObject <Node> ();

  Ptr<LrWpanNetDevice> dev0 = CreateObject<LrWpanNetDevice> ();
  Ptr<LrWpanNetDevice> dev1 = CreateObject<LrWpanNetDevice> ();
  g_device = dev1;  // Assigning dev1 to the global variable
  g_coordinatorDevice = dev0;
  dev0->SetAddress (Mac16Address ("00:01"));
  g_device ->SetAddress (Mac16Address ("00:02"));

  Ptr<SingleModelSpectrumChannel> channel = CreateObject<SingleModelSpectrumChannel> ();
  // Attach the shadowing model to the channel
  Ptr<LogDistancePropagationLossModel> propModel = CreateObject<LogDistancePropagationLossModel> ();
  propModel->SetAttribute("Exponent", DoubleValue(2));

  Ptr<RandomPropagationLossModel> randomLossModel = CreateObject<RandomPropagationLossModel>();
//Optionally configure the random variable here
 randomLossModel->SetAttribute("Variable", StringValue("ns3::UniformRandomVariable[Min=1.0|Max=3.0]"));
 channel->AddPropagationLossModel(randomLossModel);


  Ptr<ConstantSpeedPropagationDelayModel> delayModel = CreateObject<ConstantSpeedPropagationDelayModel> ();
  channel->AddPropagationLossModel (propModel);
  channel->SetPropagationDelayModel (delayModel);



// Set the default error rate
  Config::SetDefault("ns3::BurstErrorModel::ErrorRate", DoubleValue(0.12));

// Set the default values for the minimum and maximum burst duration
 Config::SetDefault("ns3::BurstErrorModel::MinBurstDuration", TimeValue(Seconds(0.1)));
 Config::SetDefault("ns3::BurstErrorModel::MaxBurstDuration", TimeValue(Seconds(1.0)));

// Create the BurstErrorModel object
 Ptr<BurstErrorModel> burstErrorModel = CreateObject<BurstErrorModel>();

// Set the error model for the device
  g_coordinatorDevice->GetObject<LrWpanNetDevice>()->GetPhy()->SetPostReceptionErrorModel(burstErrorModel);


  g_coordinatorDevice->SetChannel (channel);
  g_device->SetChannel (channel);


  // seting Mac-Max
  g_device ->GetMac()->SetMacMaxFrameRetries(MacMaxFrameRetries);

  n0->AddDevice (g_coordinatorDevice);
  n1->AddDevice (g_device );

  ///////////////// Mobility   ///////////////////////
  Ptr<ConstantPositionMobilityModel> sender0Mobility = CreateObject<ConstantPositionMobilityModel> ();
  sender0Mobility->SetPosition (Vector (0,0,0));
  g_coordinatorDevice->GetPhy ()->SetMobility (sender0Mobility);
  Ptr<ConstantPositionMobilityModel> sender1Mobility = CreateObject<ConstantPositionMobilityModel> ();

  sender1Mobility->SetPosition (Vector (25,0,0)); //10 m distance
  g_device ->GetPhy ()->SetMobility (sender1Mobility);


  /////// MAC layer Callbacks hooks/////////////

  MlmeStartConfirmCallback cb0;
  cb0 = MakeCallback (&StartConfirm);
  g_coordinatorDevice->GetMac ()->SetMlmeStartConfirmCallback (cb0);



  McpsDataConfirmCallback cb1;
  cb1 = MakeCallback(&TransEndIndication);
  g_device ->GetMac ()->SetMcpsDataConfirmCallback (cb1);

 RetransmissionCallback retransmissionCb = MakeCallback(&OnRetransmission);
 g_device->GetMac()->SetRetransmissionCallback(retransmissionCb);

  MlmeBeaconNotifyIndicationCallback cb3;
  cb3 = MakeCallback (&BeaconIndication);
  g_device ->GetMac ()->SetMlmeBeaconNotifyIndicationCallback (cb3);

  McpsDataIndicationCallback cb4;
  cb4 = MakeCallback (&DataIndication);
  g_device ->GetMac ()->SetMcpsDataIndicationCallback (cb4);

// Add the retransmission callback

  McpsDataIndicationCallback cb5;
  cb5 = MakeCallback (&DataIndicationCoordinator);
  g_coordinatorDevice->GetMac ()->SetMcpsDataIndicationCallback (cb5);

     LrWpanMacTransCostCallback txTimeCallback = MakeCallback(&TransactionTimeHandler);
    g_device->GetCsmaCa()->SetLrWpanMacTransCostCallback(txTimeCallback);

  LrWpanMacBackoffTimeCallback backoffTimeCallback = MakeCallback(&BackoffTimeHandler);
  g_device->GetCsmaCa()->SetLrWpanMacBackoffTimeCallback(backoffTimeCallback);



  g_device ->GetMac ()->SetPanId (5);
  g_device ->GetMac ()->SetAssociatedCoor (Mac16Address ("00:01"));

  ///////////////////// Start transmitting beacons from coordinator ////////////////////////

  MlmeStartRequestParams params;
  params.m_panCoor = true;
  params.m_PanId = 5;
  params.m_bcnOrd = 10; //10
  params.m_sfrmOrd = 10;
  Simulator::ScheduleWithContext (1, Seconds (1.6),
                                  &LrWpanMac::MlmeStartRequest,
                                  g_coordinatorDevice->GetMac (), params);


  Simulator::ScheduleWithContext (1, Seconds (2.0), &SendPacket, g_device );
  //Simulator::Schedule(Seconds(beaconInterval), &EndOfBeaconInterval);


  // Start the toggling process
 //ToggleRadio();
 int simulationDuration = 100;
  Simulator::Stop (Seconds (simulationDuration));
  Simulator::Run ();

  double throughput = (numReceivedPackets * 100 * 8) / totalDelay;
  double pdr = static_cast<double>(numReceivedPackets) / numSentPackets;
  //double Latency = totalDelay;
  double totaldelay = g_totalTxCost + g_totalTxbackoff;
  double Latency = totalDelay/numSentPackets;
  double EnergyPerSec=totalEnergy/simulationDuration;
  double energConsumed = g_totalTxCost*txPower + g_totalTxbackoff*sleepPower;
  NS_LOG_UNCOND ("Throughput: " << throughput << " bps");
  NS_LOG_UNCOND ("Packet Delivery Ratio (PDR): " << pdr);
  NS_LOG_UNCOND ("Latency: " << Latency << " ms");
  NS_LOG_UNCOND ("Total Energy Consumed: " << EnergyPerSec << " mJ");
  NS_LOG_UNCOND ("Total power: " << energConsumed << " mJ");
  NS_LOG_UNCOND("Total Transaction Cost: " << totaldelay << " secs");


  Simulator::Destroy ();
  return 0;
}
