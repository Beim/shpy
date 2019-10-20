import re


class PatternLog:

    def __init__(self, round, pattern_type, hash_code):
        self.round = int(round)
        self.pattern_type = pattern_type
        self.hash_code = hash_code

    @staticmethod
    def from_str(log_str):
        pattern = r'.*logPattern, round: (.*), patternType: (.*), hashCode: (.*), startRelType.*'
        match_obj = re.match(pattern, log_str)
        if not match_obj:
            return None
        return PatternLog(match_obj.group(1), match_obj.group(2), match_obj.group(3))


class Analyser:

    def __init__(self, pruning_pattern_logs, default_pattern_logs):
        self.pruning_pattern_logs = pruning_pattern_logs
        self.default_pattern_logs = default_pattern_logs
        self.joint_pruning_pattern_logs, self.joint_default_pattern_logs = self._get_joint_pattern_logs()

    def print_numerical_rel(self):
        print('生成关系数量: pruning[%s], default:[%s]' % (
            len(self.pruning_pattern_logs),
            len(self.default_pattern_logs)
        ))

    def print_auto_ratio(self):
        count = 0
        for pattern_log in self.joint_pruning_pattern_logs:
            if 'auto' in pattern_log.pattern_type:
                count += 1
        ratio = count / len(self.joint_pruning_pattern_logs)
        print('自动筛选比例：%s' % ratio)

    def print_statistics(self):
        TP, FP, FN, TN = self._get_matrix()
        ALL = TP + FP + FN + TN
        accuracy = (TP + TN) / ALL
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        print('accuracy: %s, precision: %s, recall: %s' %
              (accuracy, precision, recall))

    def _get_matrix(self):
        """
        TP: autoPruning
        FP: 0
        FN: manualMeaningless
        TN: manualLabel
        :return:
        """
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for pattern_log in self.joint_pruning_pattern_logs:
            if 'autoPruning' in pattern_log.pattern_type:
                TP += 1
            elif 'manualMeaningless' in pattern_log.pattern_type:
                FN += 1
            elif 'manualLabel' in pattern_log.pattern_type:
                TN += 1
        return TP, FP, FN, TN

    def _get_joint_pattern_logs(self):
        joint_pruning_pattern_logs = []
        joint_default_pattern_logs = []
        pruning_hash_code_set = set()
        default_hash_code_set = set()

        for pattern_log in self.pruning_pattern_logs:
            pruning_hash_code_set.add(pattern_log.hash_code)
        for pattern_log in self.default_pattern_logs:
            default_hash_code_set.add(pattern_log.hash_code)
        joint_hash_code_set = pruning_hash_code_set.intersection(default_hash_code_set)

        for pattern_log in self.pruning_pattern_logs:
            if pattern_log.hash_code in joint_hash_code_set:
                joint_pruning_pattern_logs.append(pattern_log)
        for pattern_log in self.default_pattern_logs:
            if pattern_log.hash_code in joint_hash_code_set:
                joint_default_pattern_logs.append(pattern_log)

        return joint_pruning_pattern_logs, joint_default_pattern_logs

    @staticmethod
    def build(log_path):
        pruning_pattern_logs = []
        default_pattern_logs = []
        with open('%s/pruning.txt' % log_path, 'r') as f:
            pruning_logs = f.read().split('\n')
        with open('%s/default.txt' % log_path, 'r') as f:
            default_logs = f.read().split('\n')
        for log_str in pruning_logs:
            log_obj = PatternLog.from_str(log_str)
            if log_obj:
                pruning_pattern_logs.append(log_obj)
        for log_str in default_logs:
            log_obj = PatternLog.from_str(log_str)
            if log_obj:
                default_pattern_logs.append(log_obj)
        return Analyser(pruning_pattern_logs, default_pattern_logs)



if __name__ == '__main__':
    analyser = Analyser.build('./logs')
    analyser.print_numerical_rel()
    analyser.print_auto_ratio()
    analyser.print_statistics()


