#include "mygym.h"
#include "ns3/opengym-module.h"
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

using namespace ns3;
NS_LOG_COMPONENT_DEFINE ("OpenGym");
double totalEnergy = 0;
double totalDelay;
int retries;
uint8_t MacMaxFrameRetries = 5;
double emaChannelState = 1.0; // Global variable
//const double alpha = 0.1; // Smoothing factor, adjust as needed
int numSentPackets =0;
int numReceivedPackets = 0;
uint32_t actionValue;
uint32_t  newMacMinBE;
std::deque<int> transmissionHistory;
const size_t historySize = 8; 
Ptr<LrWpanNetDevice> g_device;
Ptr<LrWpanNetDevice> g_coordinatorDevice;
Ptr<OpenGymInterface> openGymInterface;
static double g_totalTxCost = 0.0;
static double g_totalTxbackoff = 0.0;
double txTime;
double sleepEnergy = 0;
double sleepPower = 1.4;
double txPower = 57.42;

Ptr<OpenGymSpace> MyGymEnv::GetObservationSpace() {
    NS_LOG_FUNCTION(this);

    float low = 0.0;
    float high = 1.0;
    std::vector<uint32_t> shape = {3,};
    std::string dtype = TypeNameGet<uint32_t> ();                  
    // Create the OpenGymBoxSpace with the correct number of arguments
    Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace>(low, high, shape, dtype);

    NS_LOG_UNCOND("GetObservationSpace: " << space);
    return space;
}

Ptr<OpenGymSpace> MyGymEnv::GetActionSpace()
{
    NS_LOG_FUNCTION (this);

    // There are three actions in the action space
    uint32_t n_actions = 5;
    Ptr<OpenGymDiscreteSpace> space = CreateObject<OpenGymDiscreteSpace> (n_actions);

    NS_LOG_UNCOND ("GetActionSpace: " << space);
    return space;
}


bool MyGymEnv::GetGameOver()
{
  NS_LOG_FUNCTION (this);

  // Game-over condition based on the number of sent packets
  bool isGameOver = (numSentPackets >= 2000);

  NS_LOG_UNCOND ("GetGameOver: " << isGameOver);
  return isGameOver;
}

void ScheduleNextStateRead(Ptr<OpenGymInterface> openGymInterface) {
    // Notify the OpenGymInterface that the current state is ready
    openGymInterface->NotifyCurrentState();
}

bool MyGymEnv::ExecuteActions(Ptr<OpenGymDataContainer> actionContainer) {
    NS_LOG_FUNCTION(this);

    Ptr<OpenGymDiscreteContainer> discreteAction = DynamicCast<OpenGymDiscreteContainer>(actionContainer);
    actionValue = discreteAction->GetValue();

    // Map action to MacMinBE
    newMacMinBE = actionValue+2;

    NS_LOG_UNCOND("ExecuteActions: ActionValue = " << actionValue << ", MacMinBE = " << newMacMinBE);
    return true;
}
float DiscretizeEma(float emaValue) {
    const int numBins = 5; // change number of bins here
    const float binSize = 1.0f / (numBins - 1); // Size of each bin

    // Calculate the index of the bin that pdr falls into
    int binIndex = static_cast<int>(emaValue / binSize + 0.5f); // Adding 0.5f for rounding to nearest bin

    // Clamp binIndex to be within the valid range
    if (binIndex < 0) binIndex = 0;
    if (binIndex >= numBins) binIndex = numBins - 1;

    // Return the discretized value
    return binIndex * binSize;
}

Ptr<OpenGymDataContainer> MyGymEnv::GetObservation() {
    NS_LOG_FUNCTION(this);

    std::vector<uint32_t> shape = {3,}; // Or {4,} if including MacMinBE

    Ptr<OpenGymBoxContainer<float>> box = CreateObject<OpenGymBoxContainer<float>>(shape);

    float retransmissionCount = retries/7; // Replace with actual function call
    float lastTransmissionStatus = !transmissionHistory.empty() ? static_cast<float>(transmissionHistory.back()) : 0.0f;

    float discretizedEma = DiscretizeEma(emaChannelState);
    box->AddValue(discretizedEma);
    box->AddValue(retransmissionCount);
    box->AddValue(lastTransmissionStatus);
    // box->AddValue(macMinBE); // Uncomment if including MacMinBE

    NS_LOG_UNCOND("MyGetObservation: " << box);
    return box;
}

float MyGymEnv::GetReward() {
    NS_LOG_FUNCTION(this);
    float reward = 0.0;
    float lastTransmissionStatus = !transmissionHistory.empty() ? static_cast<float>(transmissionHistory.back()) : 0.0f;
   
    //float retransmissionCount = retries; 
    float discretizedEma = DiscretizeEma(emaChannelState);
    
    // Normalizing the energy state and the action value to the same scale (0 to 1)
    reward =0.1*lastTransmissionStatus-abs(4*(1-discretizedEma)-static_cast<float>(actionValue));        
    std::cout << "Reward: " << reward << std::endl;               
    NS_LOG_UNCOND("  Reward: " << reward);
    return reward;
}

//***************************************************************************
//***************************************************************************


static void SendPacket(Ptr<LrWpanNetDevice> device)
{
 
        Ptr<Packet> p = Create<Packet>(100);
        // Define and set up the MCPS data request parameters
        McpsDataRequestParams params;
        params.m_dstPanId = 5;
        params.m_srcAddrMode = SHORT_ADDR;
        params.m_dstAddrMode = SHORT_ADDR;
        params.m_dstAddr = Mac16Address("00:01");
        params.m_msduHandle = 0; // or some unique handle
        params.m_txOptions = TX_OPTION_ACK;
        device->GetMac()->McpsDataRequest(params, p);
        numSentPackets++;
        //Simulator::Schedule(Seconds(0.2), &SendPacket, device);
        
}

static void TransEndIndication (McpsDataConfirmParams params)
{

   while (transmissionHistory.size() > historySize)
    {
        transmissionHistory.pop_front();
        
    }
    
    if (params.m_status == LrWpanMcpsDataConfirmStatus::IEEE_802_15_4_SUCCESS)
    {
        numReceivedPackets++;
        Ptr<LrWpanCsmaCa> csmaCa = g_device->GetMac()->GetCsmaCa();
        uint8_t defaultMacMinBE = 3;
        csmaCa->SetMacMinBE(defaultMacMinBE);
        NS_LOG_UNCOND("MacMinBE reset to default: " << (uint32_t)defaultMacMinBE);
        retries = g_device->GetMac()->GetRetransmissionCount();
        transmissionHistory.push_back(1); 
        
     	NS_LOG_UNCOND (Simulator::Now ().GetSeconds () << " secs | Transmission successfully sent after "
                      << retries << " retries." );
        Simulator::Schedule(Seconds(0.3), &SendPacket, g_device);
        ScheduleNextStateRead(openGymInterface);
     
    }
    
    else if (params.m_status == LrWpanMcpsDataConfirmStatus::IEEE_802_15_4_NO_ACK)
    
    {
        retries = g_device->GetMac()->GetRetransmissionCount();
        NS_LOG_UNCOND (Simulator::Now ().GetSeconds () << " secs | Packet failed after "
                      << retries << " retries");
        
        transmissionHistory.push_back(0);
      
        ScheduleNextStateRead(openGymInterface);
        Simulator::Schedule(Seconds(0.3), &SendPacket, g_device);
    }
    
    int successfulTransmissions = std::count(transmissionHistory.begin(), transmissionHistory.end(), 1);
    double currentValue = static_cast<double>(successfulTransmissions) / transmissionHistory.size();
    emaChannelState = currentValue;
    //emaChannelState = (1 - alpha) * currentValue + alpha * emaChannelState;    
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
static void BeaconIndication (MlmeBeaconNotifyIndicationParams params, Ptr<Packet> p)
{
  NS_LOG_UNCOND (Simulator::Now ().GetSeconds () 
  << " secs | Received BEACON packet of size " << p->GetSize ());

}

static void DataIndication (McpsDataIndicationParams params, Ptr<Packet> p)
{
  NS_LOG_UNCOND (Simulator::Now ().GetSeconds () << " secs | Received DATA packet of size " << p->GetSize ());
}

void OnRetransmission(uint32_t retransmissionCount) {
    NS_LOG_UNCOND("Retransmission attempt number: " << retransmissionCount);
    retries=retransmissionCount;
    Ptr<LrWpanCsmaCa> csmaCa = g_device->GetMac()->GetCsmaCa();

    transmissionHistory.push_back(0);
    int successfulTransmissions = std::count(transmissionHistory.begin(), transmissionHistory.end(), 1);
    double currentValue = static_cast<double>(successfulTransmissions) / transmissionHistory.size();
    emaChannelState = currentValue;
    //emaChannelState = (1 - alpha) * currentValue + alpha * emaChannelState;
    
     //NS_LOG_UNCOND (Simulator::Now ().GetSeconds () << "  EMA: " << emaChannelState);
     ScheduleNextStateRead(openGymInterface);
     NS_LOG_UNCOND("Retransmission attempt number: " << retransmissionCount);
  
    // Apply and log the new BE
    csmaCa->SetMacMinBE(newMacMinBE);
    NS_LOG_UNCOND("New MacMinBE set to: " << (uint32_t)newMacMinBE);
    while (transmissionHistory.size() > historySize)
    {
        transmissionHistory.pop_front();
    }
}
void TransactionTimeHandler(uint32_t transactionSymbols) {
    double symbolRate = 62500; // Example: IEEE 802.15.4 standard symbol rate
    double txTime = (double)transactionSymbols / symbolRate;

    // Accumulate the transaction cost
    g_totalTxCost += txTime;

}

void BackoffTimeHandler(double backoffTime) {
    NS_LOG_UNCOND("Current Backoff Time: " << backoffTime << " secs");
    // Additional processing or accumulation of backoff times
    g_totalTxbackoff += backoffTime ;
    sleepEnergy += txPower*backoffTime;
    
}


//**********************************************************************************************
//**********************************************************************************************
int main (int argc, char *argv[])
{

  uint16_t openGymPort = 5555; // Example port number
  openGymInterface = CreateObject<OpenGymInterface>(openGymPort);
  Ptr<MyGymEnv> myGymEnv = CreateObject<MyGymEnv> ();
  myGymEnv->SetOpenGymInterface(openGymInterface);
  
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
  propModel->SetAttribute("Exponent", DoubleValue(2.0));
  Ptr<ConstantSpeedPropagationDelayModel> delayModel = CreateObject<ConstantSpeedPropagationDelayModel> ();
  
  // Set the default error rate
  Config::SetDefault("ns3::BurstErrorModel::ErrorRate", DoubleValue(0.1));

  // Set the default values for the minimum and maximum burst duration
  Config::SetDefault("ns3::BurstErrorModel::MinBurstDuration", TimeValue(Seconds(0.1)));
  Config::SetDefault("ns3::BurstErrorModel::MaxBurstDuration", TimeValue(Seconds(1.0)));

// Create the BurstErrorModel object
Ptr<BurstErrorModel> burstErrorModel = CreateObject<BurstErrorModel>();

// Set the error model for the device
 g_coordinatorDevice->GetObject<LrWpanNetDevice>()->GetPhy()->SetPostReceptionErrorModel(burstErrorModel);
  
  channel->AddPropagationLossModel (propModel);
  channel->SetPropagationDelayModel (delayModel);

  g_coordinatorDevice->SetChannel (channel);
  g_device ->SetChannel (channel);
  
  // seting Mac-Max
  g_device ->GetMac()->SetMacMaxFrameRetries(MacMaxFrameRetries);

  n0->AddDevice (g_coordinatorDevice);
  n1->AddDevice (g_device );

  ///////////////// Mobility   ///////////////////////
  Ptr<ConstantPositionMobilityModel> sender0Mobility = CreateObject<ConstantPositionMobilityModel> ();
  sender0Mobility->SetPosition (Vector (0,0,0));
  g_coordinatorDevice->GetPhy ()->SetMobility (sender0Mobility);
  Ptr<ConstantPositionMobilityModel> sender1Mobility = CreateObject<ConstantPositionMobilityModel> ();

  sender1Mobility->SetPosition (Vector (20,0,0)); //10 m distance
  g_device ->GetPhy ()->SetMobility (sender1Mobility);


  /////// MAC layer Callbacks hooks/////////////

  MlmeStartConfirmCallback cb0;
  cb0 = MakeCallback (&StartConfirm);
  g_coordinatorDevice->GetMac ()->SetMlmeStartConfirmCallback (cb0);

  LrWpanMacTransCostCallback txTimeCallback = MakeCallback(&TransactionTimeHandler);
  g_device->GetCsmaCa()->SetLrWpanMacTransCostCallback(txTimeCallback);
  
   LrWpanMacBackoffTimeCallback backoffTimeCallback = MakeCallback(&BackoffTimeHandler);
   g_device->GetCsmaCa()->SetLrWpanMacBackoffTimeCallback(backoffTimeCallback);
    
  McpsDataConfirmCallback cb1;
  cb1 = MakeCallback(&TransEndIndication);
  g_device ->GetMac ()->SetMcpsDataConfirmCallback (cb1);

  MlmeBeaconNotifyIndicationCallback cb3;
  cb3 = MakeCallback (&BeaconIndication);
  g_device ->GetMac ()->SetMlmeBeaconNotifyIndicationCallback (cb3);

  McpsDataIndicationCallback cb4;
  cb4 = MakeCallback (&DataIndication);
  g_device ->GetMac ()->SetMcpsDataIndicationCallback (cb4);

  RetransmissionCallback retransmissionCb = MakeCallback(&OnRetransmission);
  g_device->GetMac()->SetRetransmissionCallback(retransmissionCb);
  
  McpsDataIndicationCallback cb5;
  cb5 = MakeCallback (&DataIndicationCoordinator);
  g_coordinatorDevice->GetMac ()->SetMcpsDataIndicationCallback (cb5);


  g_device ->GetMac ()->SetPanId (5);
  g_device ->GetMac ()->SetAssociatedCoor (Mac16Address ("00:01"));


  MlmeStartRequestParams params;
  params.m_panCoor = true;
  params.m_PanId = 5;
  params.m_bcnOrd = 10; //10
  params.m_sfrmOrd = 10; //6
  Simulator::ScheduleWithContext (1, Seconds (1.6),
                                  &LrWpanMac::MlmeStartRequest,
                                  g_coordinatorDevice->GetMac (), params);


  Simulator::ScheduleWithContext (1, Seconds (2.0), &SendPacket, g_device );

  //double envStepTime = globalPacketDelay + 0.1 ; // for example, 0.1 seconds
     int simulationDuration = 100;
  Simulator::Stop (Seconds (simulationDuration));
 
  //Simulator::Schedule(Seconds(envStepTime), &ScheduleNextStateRead, envStepTime, openGymInterface);
  Simulator::Run ();

  openGymInterface->NotifySimulationEnd();
   double totaldelay = g_totalTxCost + g_totalTxbackoff;
  double throughput = (numReceivedPackets * 100 * 8) / totaldelay;
  double pdr = static_cast<double>(numReceivedPackets) / numSentPackets;
  //double Latency = totalDelay;
 
  double Latency = totaldelay/numSentPackets;
  double energConsumed = g_totalTxCost*txPower + g_totalTxbackoff*sleepPower;
  NS_LOG_UNCOND ("Throughput: " << throughput << " bps");
  NS_LOG_UNCOND ("Packet Delivery Ratio (PDR): " << pdr);
  NS_LOG_UNCOND ("Latency: " << Latency << " ms");
  NS_LOG_UNCOND("Total Transaction Cost: " << totaldelay << " secs"); 
   
  Simulator::Destroy ();
  NS_LOG_UNCOND ("Simulation end");
  return 0;
}
