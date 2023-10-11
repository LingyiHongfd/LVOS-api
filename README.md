# LVOS-api
APIs for LVOS (<a href="https://lingyihongfd.github.io/lvos.github.io/">[ICCV 2023] LVOS: A Benchmark for Long-term Video Object Segmentation</a>)

## Tools

We provided the tools to process LVOS datasets.

##### Convert Masks

Because we release all the annotations of validation sets, the file structure is a little different from <a href="https://youtube-vos.org/"> YouTube-VOS</a> and <a href="https://davischallenge.org//"> DAVIS</a>, which brings some inconveniences for usage. We provide the codes to convert masks into YouTube-VOS validation format, where only the first masks of each object exists. Thus, you can easily evaluate LVOS by using existing codebase, such as AOT, STCN, XMem, and so on.

To convert the masks, execute the following command.

```bash
python tools/convert_mask_for_eval.py --lvos_path /path/to/LVOS/val
```

Please replace the `/path/to/LVOS/val` with your own valid split path. Then a new folder `Annotations_convert` is created under the valid path, where the masks are in YouTube-VOS format.


After running code, file structure will be as follows:

```
{LVOS ROOT}
|-- train
    |-- JPEGImages
        |-- video1
            |-- 00000001.jpg
            |-- ...
    |-- Annotations
        |-- video1
            |-- 00000001.jpg
            |-- ...
    |-- train_meta.json
    |-- train_expression_meta.json
|-- val
    |-- JPEGImages
        |-- ...
    |-- Annotations
        |-- ...
    |-- Annotations_convert
        |-- ...
    |-- train_meta.json
    |-- train_expression_meta.json
|-- test
    |-- JPEGImages
        |-- ...
    |-- Annotations
        |-- ...
    |-- train_meta.json
    |-- train_expression_meta.json
```



## Test Scripts

For further investigation of LVOS, we provide the test script modified by ourselves.

We modified <a href="https://github.com/yoxu515/aot-benchmark"> AOT</a>, <a href="https://github.com/hkchengrex/STCN"> STCN</a>, and <a href="https://github.com/hkchengrex/XMem">XMem </a>. See more in `eval_scripts` folder. Just download the origin codes of these models, and put these files under corresponding folder. 

For AOT, we modified the config and dataloader codes, and restricted the memory length into 6 frames, which is the same as the setting in our paper. 

For XMem, we only add the dataloader codes.

For STCN, because the origin evalutation codes are time-consuming and require a large amount of GPU memory when the video is longer. We convert the STCN evaluation codes into XMem manner, and this modification saves a large number of GPU memory and speeds up the inference process.





## Evaluation

Please our <a href="https://github.com/LingyiHongfd/lvos-evaluation">evaluation toolkits</a> to assess your model's result on validation set. See this repository for more details on the usage of toolkits.

For test set, please use the <a href="https://codalab.lisn.upsaclay.fr/competitions/8767">CodaLab</a> server for convenience of evaluating your own algorithms.








