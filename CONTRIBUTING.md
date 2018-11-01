# Contributing to PocketFlow
[腾讯开源激励计划](https://opensource.tencent.com/contribution) 鼓励开发者的参与和贡献，期待你的加入。我们欢迎[report Issues](https://github.com/Tencent/PocketFlow/issues) 或者 [pull requests](https://github.com/Tencent/PocketFlow/pulls)。 在贡献代码之前请阅读以下指引。

## 问题管理
我们用 Github Issues 去跟踪 public bugs 和 feature requests。

### 查找已知的issue 优先
请查找已存在或者相类似的issue，从而保证不存在冗余。

### 新建 Issues
新建issues 时请提供详细的描述、截屏或者短视频来辅助我们定位问题

### 分支管理

有两个主分支：

1. `master` 分支
    1. **注意不要提交PR到此分支**
2. `dev` 分支. 
    1. **这是稳定的开发分支，经过完成测试后，`dev`分支的内容会在下次发布时合并到 `master`分支。**
    2. **建议提交PR到`dev`分支。**

###  Pull Requests

我们欢迎大家贡献代码来使我们的PocketFlow更加强大
代码团队会监控pull request, 我们会做相应的代码检查和测试，测试通过之后我们就会接纳PR ，但是不会立即合并到master分支。

在完成一个pr之前请做一下确认:

1. 从 `master`  fork 你自己的分支。
2. 在修改了代码之后请修改对应的文档和注释。
3. 在新建的文件中请加入licence 和copy right申明。
4. 确保一致的代码风格，可运行脚本run_pylint.sh进行一致性检查。
5. 做充分的测试。
6. 然后，你可以提交你的代码到 `dev` 分支。

## 代码协议
[BSD 3-Clause License](https://github.com/Tencent/PocketFlow/blob/master/LICENSE.TXT) 为PocketFlow的开源协议，您贡献的代码也会受此协议保护。
