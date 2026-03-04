import jieba      # 中文分词库，提供内置词典和词频数据
import random     # 随机数模块，用于生成随机切分策略
import math       # 数学模块，用于对数计算

class segmentation:
    """
    中文分词器（抽象版）
    注：切分策略均为非标准算法，仅用于演示分词歧义
    """
    
    # 类属性：定义可用的切分模式
    # 字典结构：键是字符串'1'到'5'，值是元组(英文名, 中文描述)
    MODES = {
        '1': ('shredder', '随机2-4字窗口'),
        '2': ('ghost', '强制2字滑动'),
        '3': ('quantum', '随机长度+随机跳跃'),
        '4': ('fusion', '强行合并相邻片段'),
        '5': ('random', '随机选择上述模式')
    }
    
    def __init__(self, seed=None):
        """
        构造方法：创建对象时自动调用
        seed: 随机种子，固定后每次切分结果相同（便于复现）
        """
        if seed:  # 如果传入了seed参数
            random.seed(seed)  # 设置随机种子，让随机变得可预测
        
        # 初始化jieba词典，第一次调用会加载dict.txt文件（可能需要1-2秒）
        jieba.initialize()
        
        # 获取jieba词典中所有词的总频率，用于后续计算百分比
        self.total_freq = jieba.dt.total
        
    def _get_plausibility(self, word: str) -> float:
        """
        私有方法（单下划线开头）：计算单个词在通用语料中的似真度
        word: 要查询的字符串
        return: 返回0-100的浮点数
        """
        # 从jieba词典查词频，查不到返回0
        freq = jieba.dt.FREQ.get(word, 0)
        
        if freq == 0:
            return 0.0  # 词典里没有这个词，似真度为0
        
        # 对数归一化计算：
        # 1. math.log(freq + 1) 防止log(0)报错，同时压缩高频词的差距
        # 2. 除以总频率的对数，映射到0-1区间
        # 3. 乘150放大到0-150，再用min限制在100以内
        score = min(100, (math.log(freq + 1) / math.log(self.total_freq)) * 150)
        return score
    
    def greedy_shredder(self, text: str) -> list:
        """
        贪婪碎纸策略：随机长度滑动窗口
        text: 输入字符串
        return: 返回字符串列表（切分结果）
        """
        result = []  # 初始化空列表，用于存储切分片段
        i = 0        # 索引指针，从字符串开头开始
        
        # while循环：当i小于字符串长度时继续
        while i < len(text):
            # random.randint(2, 4) 生成2、3或4的随机整数，决定窗口大小
            window = random.randint(2, 4)
            
            # text[i:i+window] 切片语法：从i开始取window个字符
            # 例如 text="北京大学", i=0, window=2 → "北京"
            result.append(text[i:i+window])
            
            # 移动指针：减去随机0或1，制造重叠效果（有时重叠1个字）
            i += window - random.randint(0, 1)
            
        return result  # 返回填充好的列表
    
    def overlap_ghost(self, text: str) -> list:
        """
        强制2字滑动：生成n-gram
        例如"北京大学" → ['北京', '京大', '大学']
        """
        # 列表推导式：遍历range(len(text) - 1)，对每个i取text[i:i+2]
        # len(text) - 1 确保i+2不越界
        return [text[i:i+2] for i in range(len(text) - 1)]
    
    def quantum_jump(self, text: str) -> list:
        """
        量子跳跃：随机长度+随机步长
        """
        result = []
        i = 0
        
        while i < len(text):
            # 随机切分长度1-5
            length = random.randint(1, 5)
            result.append(text[i:i+length])
            
            # 随机跳跃步长1到length，可能跳过字符（产生"隧穿"效果）
            i += random.randint(1, length)
            
        return result
    
    def semantic_fusion(self, text: str) -> list:
        """
        语义黏连：先粗切为2字块，再随机合并
        """
        # 第一步：将文本切成2字一块（最后一个可能只有1字）
        # range(0, len(text), 2) 从0开始，步长2，即0,2,4...
        raw = [text[i:i+2] for i in range(0, len(text), 2)]
        
        result = []
        i = 0
        
        while i < len(raw):
            # random.random() 生成0-1之间随机小数
            # 如果大于0.6（60%概率）且不是最后一个，就合并当前和下一个
            if random.random() > 0.6 and i < len(raw) - 1:
                # 字符串拼接：如'参加' + '期末' = '参加期末'
                result.append(raw[i] + raw[i+1])
                i += 2  # 跳过两个（因为用了两个）
            else:
                result.append(raw[i])
                i += 1  # 正常跳一个
                
        return result
    
    def show_help(self):
        """
        打印帮助信息
        """
        print("""
【说明】
本工具通过非标准切分策略（随机窗口、强制2-gram等）生成分词碎片，
并基于jieba词频库评估每个碎片"像不像正常词汇"。

似真度判定：
  60%+  ：常见词（如"我们"）
  20-60%：边缘词（如"楚楚"）
  0-20% ：罕见组合（如"学参"）
  0%    ：未收录（注意，乔姆斯基最喜欢拿这种东西嘲讽NLP，别让他看到了）

提示：仅供娱乐，切分结果不代表真实语义边界，你可以像语言学家定义"语言直觉"一样随意解释结果。
        """)
    
    def dissect(self, text: str, mode: str = "random") -> str:
        """
        主分析方法：根据模式切分文本并评估
        text: 输入文本
        mode: 模式英文标识符（如'shredder'）
        return: 格式化后的字符串报告
        """
        # 映射字典：将英文名映射到对应的方法
        # 注意：这里存的是方法对象（函数本身），不是字符串
        method_map = {
            'shredder': self.greedy_shredder,
            'ghost': self.overlap_ghost,
            'quantum': self.quantum_jump,
            'fusion': self.semantic_fusion
        }
        
        # 如果用户选择random（随机）
        if mode == 'random':
            # list(method_map.values()) 获取所有方法对象
            # random.choice 随机选一个
            mode_func = random.choice(list(method_map.values()))
            mode_name = 'random'
        else:
            # .get() 方法：从字典取值，不存在则返回None
            mode_func = method_map.get(mode)
            mode_name = mode
            
        if not mode_func:  # 如果mode_func是None（没找到对应方法）
            return f"错误：未知模式 '{mode}'"
        
        # 调用选中的切分函数，传入text，返回列表
        cuts = mode_func(text)
        
        # 反向查找中文名：遍历MODES找到对应的英文名的中文描述
        cn_name = mode  # 默认显示英文名
        for num, (en, cn) in self.MODES.items():  # .items()遍历字典的键值对
            if en == mode_name:  # 如果找到匹配
                cn_name = f"模式{num}：{cn}"  # f-string格式化字符串
                break  # 找到就跳出循环
        
        # 构建输出列表，最后用join合并
        lines = [
            f"【分词结果】",
            f"策略：{cn_name}",
            f"文本：{text}",
            f"{'片段':<10} {'词频':<10} {'似真度':<8} {'判定'}",  # :<10 表示左对齐占10字符
            "-" * 50
        ]
        
        # 遍历每个切分片段，计算统计信息
        for word in cuts:
            freq = jieba.dt.FREQ.get(word, 0)  # 查词频
            score = self._get_plausibility(word)  # 计算似真度
            
            # 条件判断：根据分数给出不同判定
            if score >= 60:
                diagnosis = "统计显著"
            elif score >= 20:
                diagnosis = "边缘输入"
            elif score > 0:
                diagnosis = "勉强像个词"
            else:
                diagnosis = "别让乔姆斯基看到"
            
            # 格式化：word左对齐10字符，freq左对齐10字符，score右对齐6字符保留1位小数
            lines.append(f"{word:<10} {freq:<10} {score:>6.1f}%   {diagnosis}")
        
        # "\n".join(lines) 用换行符连接列表中的所有字符串
        return "\n".join(lines)

# 当文件直接运行（不是被import）时执行以下代码
if __name__ == "__main__":
    # 创建类实例（对象），设置随机种子42
    cutter = segmentation(seed=42)
    
    print("中文分词器（抽象版）")
    print("输入 help 查看说明，quit 退出")
    
    # 无限循环，直到用户输入quit
    while True:
        print("\n可用策略：")
        # 遍历MODES字典，num是'1','2'...，(en, cn)是元组解包
        for num, (en, cn) in cutter.MODES.items():
            print(f"  [{num}] {cn}")  # f-string格式化输出
        
        # input()函数获取用户输入，strip()去掉首尾空格
        user_input = input("\n输入序号(1-5)：").strip()
        
        # 检查是否是退出命令（in 运算符检查是否在列表中）
        if user_input.lower() in ['quit', 'q']:
            print("再见。")
            break  # 跳出while循环，结束程序
        
        # 检查是否是帮助命令
        if user_input.lower() in ['help', 'h']:
            cutter.show_help()  # 调用实例方法
            continue  # continue跳过本次循环剩余代码，回到while开头
        
        # 检查输入是否在MODES的键中（'1','2','3','4','5'）
        if user_input not in cutter.MODES:
            print("请输入 1-5")
            continue
        
        # 从MODES字典获取对应的元组，再取索引0（英文名）
        # 例如 user_input='1' → cutter.MODES['1'] → ('shredder', '随机2-4字窗口') → 'shredder'
        mode = cutter.MODES[user_input][0]
        
        # 获取要切分的文本
        text = input("输入文本：").strip()
        if not text:  # 空字符串为False，not text为True
            print("文本不能为空")
            continue
        
        # 调用dissect方法，传入文本和模式，打印返回的报告
        print(cutter.dissect(text, mode))