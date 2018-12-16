# 为PocketFlow做出贡献

[腾讯开源激励计划](https://opensource.tencent.com/contribution)鼓励所有开发者的参与和贡献，我们期待你的加入。你可以报告 [Issues](https://github.com/Tencent/PocketFlow/issues) 或者提交 [Pull Requests](https://github.com/Tencent/PocketFlow/pulls)。在贡献代码之前，请阅读以下指引。

## 问题管理

我们使用 Github Issues 以收集问题和功能需求。

### 查找已有的 Issues

在创建新 Issue 之前，请先搜索是否存在已有的或者类似的 Issue ，以避免重复。

### 创建新 Issue

当创建新 Issue 时，请提供详细的描述、截屏以及/或者短视频来帮助我们定位和复现问题。

## 分支管理

目前，为简便起见，我们仅有一个分支：

1. `master` 分支：
   1. 这是稳定分支。高度稳定的版本将会被标注特定的版本号。
   2. 请向该分支提交包含问题修复或者新功能的 Pull Requests。

## Pull Requests

我们欢迎所有人向 PocketFlow 贡献代码。我们的代码团队会监控 Pull Requests, 进行相应的代码测试与检查，通过测试的 PR 将会被合并至 `master` 分支。

在提交 PR 之前，请确认:

1. 从主项目中 fork 代码
2. 与主项目保持同步
3. 在代码变动后，对应地修改注释与文档
4. 在新文件中加入协议与版权声明
5. 确保一致的代码风格（可使用 `run_pylint.sh`）
6. 充分测试你的代码
7. 向 `master` 分支发起 PR 请求

## 协议

[BSD 3-Clause License](https://github.com/Tencent/PocketFlow/blob/master/LICENSE.TXT)是PocketFlow的开源协议，你贡献的代码也会受此协议保护。
