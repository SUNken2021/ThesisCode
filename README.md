安装依赖库

```bash
pip install -r requrements.txt
```

生成数据,参数依赖于settings/interaction.json和settings/noninteraction.json

```bash
python data_generator.py
```


分割数据
```bash
python data_divider.py
```


训练模型
```bash
python train.py
```
