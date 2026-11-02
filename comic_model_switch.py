"""Three exclusive, lazy model branches for the separate PRO control area."""
MODES = ['1 普通版本', '2 真人优化', '3 社区狂野优化']

class DapaoComicModelSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'模式': (MODES, {'default': MODES[0]})},
                'optional': {f'model_{i}': ('MODEL', {'lazy': True}) for i in range(1, 4)}}
    RETURN_TYPES = ('MODEL',)
    RETURN_NAMES = ('选中模型',)
    FUNCTION = 'select'
    CATEGORY = '🍭大炮-llama-cpp'
    DESCRIPTION = '1 普通版本 · 2 真人优化 · 3 社区狂野优化；仅执行所选模型分支，不自动改变画风、色彩或采样参数。'

    @staticmethod
    def selected_key(mode):
        if mode not in MODES:
            raise ValueError('请选择 1 普通版本、2 真人优化或 3 社区狂野优化。')
        return f'model_{MODES.index(mode) + 1}'

    def check_lazy_status(self, 模式, **kwargs):
        key = self.selected_key(模式)
        return [key] if kwargs.get(key) is None else []

    def select(self, 模式, **kwargs):
        key = self.selected_key(模式)
        if kwargs.get(key) is None:
            raise ValueError(f'{模式}未连接模型，请检查对应线路。')
        return (kwargs[key],)

NODE_CLASS_MAPPINGS = {'DapaoComicModelSwitch': DapaoComicModelSwitch}
NODE_DISPLAY_NAME_MAPPINGS = {'DapaoComicModelSwitch': '🎛️ 漫画PRO模型切换@炮老师小课堂'}
