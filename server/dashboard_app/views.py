from django.shortcuts import render, redirect
from .tools_registry import TOOLS
import inspect
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd

def render_output(output):
    if isinstance(output, pd.DataFrame):
        return output.to_html(), 'table'
    elif hasattr(output, 'savefig'):
        buf = io.BytesIO()
        output.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return image_base64, 'image'
    return str(output), 'text'

def dashboard(request):
    return render(request, 'dashboard_app/dashboard.html', {'tools': TOOLS.keys()})

def tool_panel(request, tool):
    tool_factory = TOOLS.get(tool)
    if not tool_factory:
        return redirect('dashboard')

    instance = tool_factory()
    methods = [m for m in dir(instance) if callable(getattr(instance, m)) and not m.startswith("_")]

    selected_method = None
    params = {}
    output = None
    output_type = None

    if request.method == 'POST':
        method_name = request.POST.get('method')
        selected_method = method_name
        if method_name in methods:
            method = getattr(instance, method_name)
            sig = inspect.signature(method)
            params = sig.parameters
            try:
                kwargs = {
                    name: request.POST.get(name)
                    for name in params
                    if name != 'self'
                }
                raw_output = method(**kwargs)
                output, output_type = render_output(raw_output)
            except Exception as e:
                return render(request, 'dashboard_app/tool_panel.html', {
                    'tool': tool,
                    'methods': methods,
                    'error': str(e),
                    'selected_method': selected_method,
                    'params': params
                })

    return render(request, 'dashboard_app/tool_panel.html', {
        'tool': tool,
        'methods': methods,
        'selected_method': selected_method,
        'params': params,
        'output': output,
        'output_type': output_type
    })