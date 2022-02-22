docker buildx build --push --platform linux/amd64,linux/arm64 -t larsgebraad/psvwave:latest -f Dockerfile .
docker buildx build --push --platform linux/amd64,linux/arm64 -t larsgebraad/psvwave:development -f Dockerfile.dev .
