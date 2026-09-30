#!/usr/bin/env bash
set -e

APP_NAME="forex-trading-bot"
REGISTRY_USER="jcooper23"
IMAGE_TAG="${REGISTRY_USER}/${APP_NAME}:latest"

echo "=========================================================="
echo "🚀 Building & Deploying ${APP_NAME} for K3s (linux/amd64)"
echo "=========================================================="

# 1. Cross-compile the image for AMD64 K3s cluster nodes
echo "📦 Step 1: Cross-compiling Docker image (--platform linux/amd64)..."
docker build --platform linux/amd64 -t "${IMAGE_TAG}" .

# 2. Push to Docker Hub
echo "⬆️ Step 2: Pushing image to Docker Hub (${IMAGE_TAG})..."
docker push "${IMAGE_TAG}"

# 3. Output instructions
echo "=========================================================="
echo "✅ Image successfully built and pushed: ${IMAGE_TAG}"
echo "=========================================================="
echo "Next Steps in Portainer / K3s:"
echo "1. Ensure 'forex-bot-secrets' is configured in Portainer:"
echo "   - OANDA_ACCESS_TOKEN"
echo "   - OANDA_ACCOUNT_ID"
echo "   - OANDA_ENV (practice or live)"
echo "   - POSTGRES_PASSWORD"
echo "   - DAILY_PROFIT_TARGET (default: 25.0)"
echo "   - MAX_DAILY_LOSS (default: 25.0)"
echo "   - IGNORE_SESSION_FILTER (default: false)"
echo ""
echo "2. Apply manifests to your cluster:"
echo "   kubectl apply -f k8s/postgres.yaml"
echo "   kubectl apply -f k8s/redis.yaml"
echo "   kubectl apply -f k8s/pvc.yaml"
echo "   kubectl apply -f k8s/deployment.yaml"
echo "   kubectl apply -f k8s/service.yaml"
echo ""
echo "3. Monitor deployment status:"
echo "   kubectl rollout status deployment/${APP_NAME}"
echo "   kubectl logs -f -l app=${APP_NAME}"
echo "=========================================================="
