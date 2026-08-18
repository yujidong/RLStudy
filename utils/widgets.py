"""ipywidgets 工厂函数与常用交互组件。

设计原则：返回可独立复用的小部件 + 一个 :func:`make_interactive` 帮助
函数把绘图函数与参数滑块绑定。这样 notebook 里只需要写::

    w = make_interactive(plot_curve, params={'eps': (0.1, 0, 1, 0.01), ...})
    display(w['ui'])
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

from IPython.display import display

try:
    import ipywidgets as widgets
except ImportError:
    widgets = None  # 软依赖：未安装时 make_interactive 降级为静态图（见下方）


def _require_widgets():
    """工厂函数的统一入口：ipywidgets 缺失时给出可读的报错。"""
    if widgets is None:
        raise ImportError(
            "这个组件需要 ipywidgets：pip install ipywidgets 后重启 kernel 再试。"
            "（make_interactive 有静态降级，不需要手动处理）"
        )
    return widgets


__all__ = [
    "make_slider",
    "make_dropdown",
    "make_checkbox",
    "make_int_slider",
    "make_run_button",
    "make_interactive",
    "vbox",
    "hbox",
]


def make_slider(
    name: str,
    default: float = 0.5,
    min: float = 0.0,
    max: float = 1.0,
    step: float = 0.01,
    description: Optional[str] = None,
    readout_format: str = ".3f",
    style: Optional[dict] = None,
) -> widgets.FloatSlider:
    """一个浮点滑块。"""
    _require_widgets()
    s = widgets.FloatSlider(
        value=default,
        min=min,
        max=max,
        step=step,
        description=description or name,
        continuous_update=False,
        readout_format=readout_format,
        style={"description_width": "initial"} if style is None else style,
    )
    return s


def make_int_slider(
    name: str,
    default: int = 10,
    min: int = 0,
    max: int = 100,
    step: int = 1,
    description: Optional[str] = None,
) -> widgets.IntSlider:
    _require_widgets()
    return widgets.IntSlider(
        value=default,
        min=min,
        max=max,
        step=step,
        description=description or name,
        continuous_update=False,
        style={"description_width": "initial"},
    )


def make_dropdown(
    options: Sequence,
    value: Any = None,
    description: str = "选择",
) -> widgets.Dropdown:
    _require_widgets()
    return widgets.Dropdown(
        options=list(options),
        value=value if value is not None else options[0],
        description=description,
        style={"description_width": "initial"},
    )


def make_checkbox(
    value: bool = False, description: str = ""
) -> widgets.Checkbox:
    _require_widgets()
    return widgets.Checkbox(value=value, description=description, indent=False)


def make_run_button(label: str = "运行") -> widgets.Button:
    _require_widgets()
    return widgets.Button(description=label, button_style="primary")


def make_interactive(
    plot_fn: Callable[..., Any],
    params: Dict[str, tuple],
    layout: str = "vbox",
    show_output: bool = True,
) -> Dict[str, Any]:
    """把 plot_fn 与若干滑块绑定，返回 widgets 字典。

    Parameters
    ----------
    plot_fn : 每个参数都会作为关键字参数传入
    params : ``{name: (default, min, max, step)}``（浮点）或
             ``{name: (default, min, max, step)}``（int 自动检测：当 step 为 int 且
             default 为 int 时，使用 IntSlider）
    layout : 'vbox' | 'hbox'，控制滑块组的排版

    Returns
    -------
    dict，包含 'controls', 'output', 'ui', 'update' 等字段

    Notes
    -----
    ipywidgets 未安装时自动降级：用默认参数调用一次 ``plot_fn`` 画静态图，
    并在 'update' 里保持同样的语义（每次调用重画）。
    """
    if widgets is None:
        # 静态降级：不用滑块，用默认参数画一次 + 提示安装方法
        print("[静态降级] ipywidgets 未安装：已用默认参数画出静态图；"
              "pip install ipywidgets 并重启 kernel 后可交互调参。")
        defaults = {name: spec[0] for name, spec in params.items()}
        plot_fn(**defaults)

        def _update_static(**kwargs):
            plot_fn(**{**defaults, **kwargs})

        return {
            "controls": {}, "output": None, "ui": None,
            "update": _update_static, "interactive": None,
        }

    controls = {}
    for name, spec in params.items():
        default, lo, hi, step = spec
        if isinstance(default, int) and isinstance(step, int):
            controls[name] = make_int_slider(name, default=default, min=lo, max=hi, step=step)
        else:
            controls[name] = make_slider(name, default=default, min=lo, max=hi, step=step)

    out = widgets.Output()

    def _update(**kwargs):
        with out:
            out.clear_output(wait=True)
            plot_fn(**kwargs)

    interactive = widgets.interactive(_update, **controls)
    # interactive 默认是 VBox(Output, VBox(controls))，我们想要反过来
    if layout == "hbox":
        ui = widgets.HBox(list(controls.values()))
    else:
        ui = widgets.VBox(list(controls.values()))

    # 触发首次渲染
    _update(**{k: w.value for k, w in controls.items()})

    result = {"controls": controls, "output": out, "ui": ui, "update": _update, "interactive": interactive}
    if show_output:
        display(ui)
        display(out)
    return result


def vbox(*items) -> widgets.VBox:
    _require_widgets()
    return widgets.VBox(list(items))


def hbox(*items) -> widgets.HBox:
    _require_widgets()
    return widgets.HBox(list(items))


# -----------------------------------------------------------------------------
# 通用降级：notebook 后端不支持 widget 时给出可读错误
# -----------------------------------------------------------------------------
def ensure_widget_backend() -> bool:
    """检测 matplotlib 是否在交互后端，否则提示用户。

    在 notebook 第一个 widget 单元调用一次即可。
    """
    try:
        import matplotlib

        backend = matplotlib.get_backend().lower()
        if "widget" in backend or "ipympl" in backend or "inline" in backend or "notebook" in backend:
            return True
        # 尝试自动切换
        try:
            from IPython import get_ipython

            ip = get_ipython()
            if ip is not None:
                ip.run_line_magic("matplotlib", "widget")
                return True
        except Exception:
            return False
    except Exception:
        return False
    return False
