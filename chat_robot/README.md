# 目录结构
```
.
├── data  # 下载数据
│   └── movie-corpus
│       ├── conversations.json
│       ├── corpus.json
│       ├── formatted_movie_lines.txt
│       ├── index.json
│       ├── speakers.json
│       └── utterances.jsonl
├── list.txt  # tree>list.txt
├── models.py  # 定义模型
├── process.py  # 训练步骤
├── prepare.py  # 加载和预处理数据, 运行我们的模型
├── utils.py  # 加载和预处理数据
└── voc.py  # 加载和预处理数据