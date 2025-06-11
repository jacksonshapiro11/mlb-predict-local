#!/bin/bash

# Set region and variables
REGION=us-east-1
AMI_ID=ami-0e86e20dae3bb6ce4  # Ubuntu 22.04 LTS in us-east-1

# Get existing infrastructure IDs
VPC_ID=$(aws ec2 describe-vpcs \
  --filters "Name=isDefault,Values=true" \
  --region "$REGION" \
  --query 'Vpcs[0].VpcId' \
  --output text)

SUBNET_ID=$(aws ec2 describe-subnets \
  --region "$REGION" \
  --filters "Name=vpc-id,Values=$VPC_ID" \
  --query 'Subnets[0].SubnetId' \
  --output text)

SG_ID=$(aws ec2 describe-security-groups \
  --group-names mlb_gpu_sg \
  --region "$REGION" \
  --query 'SecurityGroups[0].GroupId' \
  --output text)

echo "Using AMI ID: $AMI_ID"
echo "VPC ID: $VPC_ID"
echo "Subnet ID: $SUBNET_ID" 
echo "Security Group ID: $SG_ID"

# Launch spot instance
echo "Launching spot instance..."
aws ec2 request-spot-instances \
  --region "$REGION" \
  --spot-price "0.30" \
  --instance-count 1 \
  --type "one-time" \
  --launch-specification "{
    \"ImageId\":\"${AMI_ID}\",
    \"InstanceType\":\"g4dn.xlarge\",
    \"KeyName\":\"mlb_gpu_key\",
    \"SubnetId\":\"${SUBNET_ID}\",
    \"SecurityGroupIds\":[\"${SG_ID}\"],
    \"BlockDeviceMappings\": [
      {\"DeviceName\":\"/dev/sda1\",
       \"Ebs\":{\"VolumeSize\":100,\"VolumeType\":\"gp3\"}}
    ]
  }"

echo "Waiting for instance to launch..."
sleep 60

# Get instance details
INST_ID=$(aws ec2 describe-instances \
  --filters "Name=instance-type,Values=g4dn.xlarge" \
            "Name=instance-state-name,Values=running" \
            "Name=key-name,Values=mlb_gpu_key" \
  --region "$REGION" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text)

if [ "$INST_ID" != "null" ] && [ "$INST_ID" != "" ]; then
    echo "Spot Instance ID: $INST_ID"
    
    PUB_IP=$(aws ec2 describe-instances \
      --instance-ids "$INST_ID" \
      --region "$REGION" \
      --query 'Reservations[0].Instances[0].PublicIpAddress' \
      --output text)
    
    echo "Public IP: $PUB_IP"
    echo ""
    echo "To connect:"
    echo "ssh -i ~/.ssh/mlb_gpu.pem ubuntu@$PUB_IP"
else
    echo "Instance not ready yet. Check manually with:"
    echo "aws ec2 describe-instances --region $REGION"
fi 