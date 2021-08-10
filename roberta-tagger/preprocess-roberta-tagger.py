import Levenshtein
import transformers
from tqdm import tqdm
import argparse
import json


def convert_tokens_to_input(string1, string2, tokenizer):
    str1 = tokenizer(string1)
    str2 = tokenizer(string2)
    tokens = list(set(str1 + str2))
    # to assure we get an independent character for every individual token in each sentence
    characters = list(
        set(
            "1234567890-qwertyuiopasdfghjklzxcvbQWERTYUIOOASDFGHJKLZXCVBNMnm,./;[]-+_=|}{><?!!@#$%^&*()iœ∑´®†\¨ˆøπ¬˚∆˙©ƒ∂ßåååååΩ≈ç√∫˜µ≤øˆ˙©ƒ¨∂®ß´∑ß´\†∂®¨あグノシーは、ネット上に存在する様々な情報を独自のアルゴリズムで収集し、評価付けを行い、ユーザーに届ける情報キュレーションサービスである。CEOの福島は、「ユーザーが面白いと思うコンテンツを、ニュースに限らず配信していった結果、自然とエンターテインメント性の強いメディアになった」と述べている†ƒ√˚©電気通信事業者（でんきつうしんじぎょうしゃ）とは、一般に固定電話や携帯電話等の電気通信サービスを提供する会社の総称。「音声やデータを運ぶ」というところから通信キャリア（または単にキャリア）や通信回線事業者（または単に回線事業者）と呼ばれることもある。携帯電話専業の会社については携帯会社と呼ぶことが多いがDocomoなどの携帯電話回線会社とAppleなどの携帯電話製造会社との混同されるため呼ばれ方が変わりつつある。回線事業者または回線会社として扱われる。∆˙˚∫∆…√©∆ç®™£∞¢§¶•ªªªª¸˛Ç◊ı˜Â¯Â¯˘¿ÆÚÒÔÓ˝ÏÎÍÅÅÅÅŒ„´‰ˇÁ¨ˆØ∏∏∏∏””’»±—‚·°‡ﬂﬁ›‹⁄`乙 了 又 与 及 丈 刃 凡 勺 互 弔 井 升 丹 乏 匁 屯 介 冗 凶 刈 匹 厄 双 孔 幻 斗 斤 且 丙 甲 凸 丘 斥 仙 凹 召 巨 占 囚 奴 尼 巧 払 汁 玄 甘 矛 込 弐 朱 吏 劣 充 妄 企 仰 伐 伏 刑 旬 旨 匠 叫 吐 吉 如 妃 尽 帆 忙 扱 朽 朴 汚 汗 江 壮 缶 肌 舟 芋 芝 巡 迅 亜 更 寿 励 含 佐 伺 伸 但 伯 伴 呉 克 却 吟 吹 呈 壱 坑 坊 妊 妨 妙 肖 尿 尾 岐 攻 忌 床 廷 忍 戒 戻 抗 抄 択 把 抜 扶 抑 杉 沖 沢 沈 没 妥 狂 秀 肝 即 芳 辛 迎 邦 岳 奉 享 盲 依 佳 侍 侮 併 免 刺 劾 卓 叔 坪 奇 奔 姓 宜 尚 屈 岬 弦 征 彼 怪 怖 肩 房 押 拐 拒 拠 拘 拙 拓 抽 抵 拍 披 抱 抹 昆 昇 枢 析 杯 枠 欧 肯 殴 況 沼 泥 泊 泌 沸 泡 炎 炊 炉 邪 祈 祉 突 肢 肪 到 茎 苗 茂 迭 迫 邸 阻 附 斉 甚 帥 衷 幽 為 盾 卑 哀 亭 帝 侯 俊 侵 促 俗 盆 冠 削 勅 貞 卸 厘 怠 叙 咲 垣 契 姻 孤 封 峡 峠 弧 悔 恒 恨 怒 威 括 挟 拷 挑 施 是 冒 架 枯 柄 柳 皆 洪 浄 津 洞 牲 狭 狩 珍 某 疫 柔 砕 窃 糾 耐 胎 胆 胞 臭 荒 荘 虐 訂 赴 軌 逃 郊 郎 香 剛 衰 畝 恋 倹 倒 倣 俸 倫 翁 兼 准 凍 剣 剖 脅 匿 栽 索 桑 唆 哲 埋 娯 娠 姫 娘 宴 宰 宵 峰 貢 唐 徐 悦 恐 恭 恵 悟 悩 扇 振 捜 挿 捕 敏 核 桟 栓 桃 殊 殉 浦 浸 泰 浜 浮 涙 浪 烈 畜 珠 畔 疾 症 疲 眠 砲 祥 称 租 秩 粋 紛 紡 紋 耗 恥 脂 朕"
        )
    )
    st1 = "".join([characters[tokens.index(x)] for x in str1])
    st2 = "".join([characters[tokens.index(x)] for x in str2])
    output_list = ["KEEP"] * len(str1)
    output = Levenshtein.editops(st1, st2)
    #     print(output,"\n",str1,"\n",str2)
    indices = [i[1] for i in output if i[0] != "insert"]
    other_indices = [i[2] for i in output if (i[0] != "insert" and i[0] != "delete")]
    delete_indices = [i[1] for i in output if i[0] == "delete"]

    new_str2 = {}
    for something in output:
        if something[0] != "delete":
            try:
                new_str2[something[1]].append(str2[something[2]])
            except:
                new_str2[something[1]] = [str2[something[2]]]
    #     print(new_str2)
    #     print(len(str2))
    m2f = [str2[min(i, len(str2) - 1)] for i in other_indices]
    for x in indices:
        try:
            output_list[x] = "MASK"
        except:
            pass
    for x in delete_indices:
        try:
            output_list[x] = "DELETE"
        except:
            pass
    new_list = output_list.copy()
    for i, j in enumerate(output_list):
        if j == "KEEP":
            new_list[i] = str1[i]
    mask2fill = []
    m2f_idx = 0
    previous_i = 1e2
    #     for i,j in enumerate(output_list):
    #         if j == "MASK":
    # #             print(i)
    #             if  i - previous_i >= 2:
    #                 mask2fill.append("SEP")
    # #                 print(previous_i)
    #                 current_idx = i
    #             mask2fill.append(m2f[m2f_idx])
    #             previous_i = i
    #             m2f_idx += 1
    former_key = 0
    for j, i in enumerate(sorted(new_str2.keys())):
        sep = ["</s>"] if (i - former_key > 1 and j != 0) else []
        mask2fill.extend(sep + new_str2[i])
        former_key = i
    mask2fill = mask2fill
    return string1, string2, output_list, mask2fill


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-parallel-data-file", type=str, metavar="STR")
    parser.add_argument("--mask-threshold", type=float)
    parser.add_argument("--output-path", type=str, metavar="STR")
    args = parser.parse_args()

    tokenizer = transformers.RobertaTokenizer.from_pretrained("roberta-base")

    with open(args.path_to_parallel_data_file, "r") as f:
        file = [i.strip() for i in f.readlines() if i.strip() != ""]
        input_file = [line.split("\t")[0] for line in file if "\t" in line]
        output_file = [line.split("\t")[1] for line in file if "\t" in line]
        print(len(output_file))

    print(output_file[0])
    print(input_file[0])
    assert len(output_file) == len(input_file)

    out_list = []
    for index in tqdm(range(len(output_file))):
        x = convert_tokens_to_input(
            input_file[index], output_file[index], tokenizer.tokenize
        )
        if x[2].count("MASK") + x[2].count("DELETE") > int(
            args.mask_threshold * (len(x[2]))
        ):
            continue
        else:
            out_list.append(x)

    with open(args.output_path, "w") as f:
        new_dict = []
        for i, j in enumerate(out_list):
            new_dict.append(
                {
                    "source": j[0],
                    "true_target": j[1],
                    "label": " ".join(j[2]),
                    "target": " ".join(j[3]).replace("Ġ", ""),
                }
            )
        f.write("\n".join([json.dumps(i) for i in new_dict]))
