#!/usr/bin/env bash
set -e

APP_NAME="forex-trading-bot"
REGISTRY_USER="jcooper23"
IMAGE_TAG="${REGISTRY_USER}/${APP_NAME}:latest"
K8S_DIR="k8s"
ENV_FILE=".env"
SECRET_NAME="forex-bot-secrets"
NAMESPACE="default"

# KUBECONFIG should point to your K3s cluster config via Tailscale
# e.g., export KUBECONFIG=~/.kube/config-k3s
if [ -z "${KUBECONFIG}" ] && [ ! -f "${HOME}/.kube/config" ]; then
    echo "⚠️  No KUBECONFIG set and no default kubeconfig found."
    echo "   Set KUBECONFIG to your K3s cluster config and retry."
    exit 1
fi

echo "=========================================================="
echo "🚀 Building & Deploying ${APP_NAME} for K3s (linux/amd64)"
echo "=========================================================="

# 1. Cross-compile the image for AMD64 K3s cluster nodes
echo "📦 Step 1: Cross-compiling Docker image (--platform linux/amd64)..."
docker build --platform linux/amd64 -t "${IMAGE_TAG}" .

# 2. Push to Docker Hub
echo "⬆️  Step 2: Pushing image to Docker Hub (${IMAGE_TAG})..."
docker push "${IMAGE_TAG}"

# 3. Create or update K8s secret from .env file
echo "🔐 Step 3: Syncing secrets from ${ENV_FILE} to K8s secret '${SECRET_NAME}'..."

if [ ! -f "${ENV_FILE}" ]; then
    echo "   ⚠️  No ${ENV_FILE} found. Skipping secret creation."
    echo "   Ensure '${SECRET_NAME}' exists in the cluster or create it manually."
else
    # Read values from .env (skip comments and blank lines)
    OANDA_ACCESS_TOKEN=$(grep -E '^OANDA_ACCESS_TOKEN=' "${ENV_FILE}" | cut -d'=' -f2-)
    OANDA_ACCOUNT_ID=$(grep -E '^OANDA_ACCOUNT_ID=' "${ENV_FILE}" | cut -d'=' -f2-)
    OANDA_ENV=$(grep -E '^OANDA_ENV=' "${ENV_FILE}" | cut -d'=' -f2- || echo "practice")
    POSTGRES_PASSWORD=$(grep -E '^POSTGRES_PASSWORD=' "${ENV_FILE}" | cut -d'=' -f2- || echo "forex-secure-pw")

    # Optional config values (use defaults if not in .env)
    DAILY_PROFIT_TARGET=$(grep -E '^DAILY_PROFIT_TARGET=' "${ENV_FILE}" | cut -d'=' -f2- || echo "25.0")
    MAX_DAILY_LOSS=$(grep -E '^MAX_DAILY_LOSS=' "${ENV_FILE}" | cut -d'=' -f2- || echo "25.0")
    IGNORE_SESSION_FILTER=$(grep -E '^IGNORE_SESSION_FILTER=' "${ENV_FILE}" | cut -d'=' -f2- || echo "false")

    # Set defaults for anything missing
    OANDA_ENV="${OANDA_ENV:-practice}"
    POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-forex-secure-pw}"
    DAILY_PROFIT_TARGET="${DAILY_PROFIT_TARGET:-25.0}"
    MAX_DAILY_LOSS="${MAX_DAILY_LOSS:-25.0}"
    IGNORE_SESSION_FILTER="${IGNORE_SESSION_FILTER:-false}"

    # Delete existing secret (if any) and recreate — kubectl apply doesn't merge secrets cleanly
    kubectl delete secret "${SECRET_NAME}" -n "${NAMESPACE}" --ignore-not-found
    kubectl create secret generic "${SECRET_NAME}" -n "${NAMESPACE}" \
        --from-literal=OANDA_ACCESS_TOKEN="${OANDA_ACCESS_TOKEN}" \
        --from-literal=OANDA_ACCOUNT_ID="${OANDA_ACCOUNT_ID}" \
        --from-literal=OANDA_ENV="${OANDA_ENV}" \
        --from-literal=POSTGRES_PASSWORD="${POSTGRES_PASSWORD}" \
        --from-literal=DAILY_PROFIT_TARGET="${DAILY_PROFIT_TARGET}" \
        --from-literal=MAX_DAILY_LOSS="${MAX_DAILY_LOSS}" \
        --from-literal=IGNORE_SESSION_FILTER="${IGNORE_SESSION_FILTER}"
    echo "   ✅ Secret '${SECRET_NAME}' created/updated."
fi

# 4. Apply K8s manifests in dependency order
echo "📋 Step 4: Applying K8s manifests..."

echo "   Applying PVC..."
kubectl apply -f "${K8S_DIR}/pvc.yaml" -n "${NAMESPACE}"

echo "   Applying PostgreSQL..."
kubectl apply -f "${K8S_DIR}/postgres.yaml" -n "${NAMESPACE}"

echo "   Applying Redis..."
kubectl apply -f "${K8S_DIR}/redis.yaml" -n "${NAMESPACE}"

echo "   Applying Bot Deployment..."
kubectl apply -f "${K8S_DIR}/deployment.yaml" -n "${NAMESPACE}"

echo "   Applying Service..."
kubectl apply -f "${K8S_DIR}/service.yaml" -n "${NAMESPACE}"

# 5. Rolling restart to pull the new image
echo "🔄 Step 5: Rolling restart to pull latest image..."
kubectl rollout restart deployment/${APP_NAME} -n "${NAMESPACE}"

# 6. Wait for rollout
echo "⏳ Step 6: Waiting for rollout to complete..."
kubectl rollout status deployment/${APP_NAME} -n "${NAMESPACE}" --timeout=120s

# 7. Show status
echo ""
echo "=========================================================="
echo "✅ Deployment complete: ${IMAGE_TAG}"
echo "=========================================================="
echo ""
kubectl get pods -l app=${APP_NAME} -n "${NAMESPACE}" -o wide
echo ""
echo "📊 Monitor logs:  kubectl logs -f -l app=${APP_NAME}"
echo "🏥 Health check:  curl http://<node-ip>:<nodeport>/health"
echo "=========================================================="
