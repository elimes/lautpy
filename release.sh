#!/bin/bash
# release.sh - 更新 .data/VERSION（唯一版本源）并打 tag 触发发布
# pyproject.toml 的版本号由 setuptools 动态读取 .data/VERSION，无需手动同步

set -e  # 出错立即退出

VERSION=$1
if [ -z "$VERSION" ]; then
  echo "Usage: $0 <version>" >&2
  echo "Example: $0 0.1.1" >&2
  exit 1
fi

echo "🚀 Releasing version v$VERSION..."

# 1. 切换并同步 main 分支
git checkout main
git pull origin main --tags

# 2. 更新唯一权威版本文件
echo "$VERSION" > .data/VERSION
echo "✅ Updated .data/VERSION"

# 3. 提交并打 tag（push tag 触发 GitHub Actions 发布到 PyPI）
git add .data/VERSION
git commit -m "chore: release v$VERSION"
git tag "v$VERSION"

# 4. 推送
git push origin main
git push origin "v$VERSION"

echo "🎉 Release v$VERSION triggered! Check GitHub Actions."
