import torch
import torch._dynamo.config
from torch._inductor.utils import run_and_get_triton_code
from torch.testing import FileCheck
from torch._dynamo import disable

# TORCH_LOGS="+dynamo,aot,inductor" TORCH_COMPILE_DEBUG=1 python test/test_cross_graph.py


class TestSubmodule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sub_weight = torch.nn.Parameter(torch.randn(4, 4))

    def forward(self, inp):
        return torch.add(self.sub_weight, inp)


@disable()
def g1_mutation_tuple(d, e):
    d.relu_()
    return d, e

@disable()
def g1_mutation_tensor(d, e):
    d.relu_()
    return d + e

@disable()
def g2(a, b):
    return torch.cat(torch.chunk(a * b, 2))

global_a = torch.randn(4, 4)

@disable()
def g2_read_global_var(a, b):
    return torch.cat(torch.chunk(a * b * global_a, 2))


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(1))  # torch.randn(4, 4))
        self.register_buffer('buf', torch.randn(1))  # torch.randn(4, 4))
        self.submod = TestSubmodule()

    @disable()
    def f_read_param_mutate_param(self, c):
        self.buf.relu_()
        return c * c * self.weight

    def forward(self, x, y):
        x.relu_()
        self.buf.relu_()
        y = torch.cat(torch.chunk(y, 2))
        return torch.relu(x) + g1_mutation_tuple(x, y)[0] \
            + g1_mutation_tensor(x, x) \
            + g2(x, x) \
            + g2(x, y) \
            + g2_read_global_var(x, y) \
            + self.f_read_param_mutate_param(x) \
            + torch.tanh(self.weight) \
            + self.buf

"""
Compile log (all graphs): https://gist.github.com/yf225/1bdc25dd0f7fee31f83cb99f486dfc5f

FuncReadWrite(fn_name='__compiled_fn_0',
              reads={'var_4', 'var_2', 'var_3'},
              mutations={'var_2', 'var_4'},
              compiled_fn=<function aot_module_simplified.<locals>.forward at 0x7fcb8a9ce3e0>,
              input_index_to_global_var_name={0: ['var_2'], 1: ['var_3']},
              output_index_to_global_var_name={0: ['var_5'], 1: ['var_6']},
              outputs={'var_5', 'var_6'},
              aliases={'L_x_': {'l_x_'},
                       'L_y_': {'l_y_'},
                       'chunk': {'getitem', 'getitem_1'},
                       'getitem': {'cat_1'},
                       'getitem_1': {'cat_1'},
                       'l__self___buf': {'relu__1'},
                       'l_x_': {'relu_'},
                       'l_y_': {'chunk'}},
              nominal_inputs=['L_x_', 'L_y_'],
              nominal_param_reads={'l__self___buf'},
              nominal_param_to_actual_param={'l__self___buf': 'L__self___buf'},
              nominal_mutations={'l__self___buf', 'l_x_'},
              reads_data_ptr=set(),
              mutations_data_ptr=set())
FuncReadWrite(fn_name='__eager_fn_7',
              reads={'var_2'},
              mutations={'var_2'},
              compiled_fn=None,
              input_index_to_global_var_name={},
              output_index_to_global_var_name={},
              outputs=set(),
              aliases={},
              nominal_inputs=[],
              nominal_param_reads=set(),
              nominal_param_to_actual_param={},
              nominal_mutations=set(),
              reads_data_ptr={116433664},
              mutations_data_ptr={116433664})
FuncReadWrite(fn_name='__compiled_fn_8',
              reads={'var_2', 'var_5'},
              mutations=set(),
              compiled_fn=<function aot_module_simplified.<locals>.forward at 0x7fcb89a17060>,
              input_index_to_global_var_name={0: ['var_5'], 1: ['var_2']},
              output_index_to_global_var_name={0: ['var_10']},
              outputs={'var_10'},
              aliases={'L_stack0_': {'l_stack0_'},
                       'L_stack1_0_': {'l_stack1_0_'}},
              nominal_inputs=['L_stack0_', 'L_stack1_0_'],
              nominal_param_reads=set(),
              nominal_param_to_actual_param={},
              nominal_mutations=set(),
              reads_data_ptr=set(),
              mutations_data_ptr=set())
FuncReadWrite(fn_name='__eager_fn_11',
              reads={'var_2'},
              mutations={'var_2'},
              compiled_fn=None,
              input_index_to_global_var_name={},
              output_index_to_global_var_name={},
              outputs=set(),
              aliases={},
              nominal_inputs=[],
              nominal_param_reads=set(),
              nominal_param_to_actual_param={},
              nominal_mutations=set(),
              reads_data_ptr={116433664},
              mutations_data_ptr={116433664})
FuncReadWrite(fn_name='__compiled_fn_12',
              reads={'var_14', 'var_10'},
              mutations=set(),
              compiled_fn=<function aot_module_simplified.<locals>.forward at 0x7fcb89d432e0>,
              input_index_to_global_var_name={0: ['var_10'], 1: ['var_14']},
              output_index_to_global_var_name={0: ['var_15']},
              outputs={'var_15'},
              aliases={'L_stack0_': {'l_stack0_'}, 'L_stack1_': {'l_stack1_'}},
              nominal_inputs=['L_stack0_', 'L_stack1_'],
              nominal_param_reads=set(),
              nominal_param_to_actual_param={},
              nominal_mutations=set(),
              reads_data_ptr=set(),
              mutations_data_ptr=set())
FuncReadWrite(fn_name='__eager_fn_16',
              reads={'var_2'},
              mutations=set(),
              compiled_fn=None,
              input_index_to_global_var_name={},
              output_index_to_global_var_name={},
              outputs=set(),
              aliases={},
              nominal_inputs=[],
              nominal_param_reads=set(),
              nominal_param_to_actual_param={},
              nominal_mutations=set(),
              reads_data_ptr={116433664, 1484354048, 1484354080},
              mutations_data_ptr=set())
FuncReadWrite(fn_name='__compiled_fn_17',
              reads={'var_15', 'var_19'},
              mutations=set(),
              compiled_fn=<function aot_module_simplified.<locals>.forward at 0x7fcb89c818a0>,
              input_index_to_global_var_name={0: ['var_15'], 1: ['var_19']},
              output_index_to_global_var_name={0: ['var_20']},
              outputs={'var_20'},
              aliases={'L_stack0_': {'l_stack0_'}, 'L_stack1_': {'l_stack1_'}},
              nominal_inputs=['L_stack0_', 'L_stack1_'],
              nominal_param_reads=set(),
              nominal_param_to_actual_param={},
              nominal_mutations=set(),
              reads_data_ptr=set(),
              mutations_data_ptr=set())
FuncReadWrite(fn_name='__eager_fn_21',
              reads={'var_2', 'var_6'},
              mutations=set(),
              compiled_fn=None,
              input_index_to_global_var_name={},
              output_index_to_global_var_name={},
              outputs=set(),
              aliases={},
              nominal_inputs=[],
              nominal_param_reads=set(),
              nominal_param_to_actual_param={},
              nominal_mutations=set(),
              reads_data_ptr={116433664, 115281536, 115281568, 1482624192},
              mutations_data_ptr=set())
FuncReadWrite(fn_name='__compiled_fn_22',
              reads={'var_24', 'var_20'},
              mutations=set(),
              compiled_fn=<function aot_module_simplified.<locals>.forward at 0x7fcb89d40cc0>,
              input_index_to_global_var_name={0: ['var_20'], 1: ['var_24']},
              output_index_to_global_var_name={0: ['var_25']},
              outputs={'var_25'},
              aliases={'L_stack0_': {'l_stack0_'}, 'L_stack1_': {'l_stack1_'}},
              nominal_inputs=['L_stack0_', 'L_stack1_'],
              nominal_param_reads=set(),
              nominal_param_to_actual_param={},
              nominal_mutations=set(),
              reads_data_ptr=set(),
              mutations_data_ptr=set())
FuncReadWrite(fn_name='__eager_fn_26',
              reads={'var_2', 'var_6'},
              mutations=set(),
              compiled_fn=None,
              input_index_to_global_var_name={},
              output_index_to_global_var_name={},
              outputs=set(),
              aliases={},
              nominal_inputs=[],
              nominal_param_reads=set(),
              nominal_param_to_actual_param={},
              nominal_mutations=set(),
              reads_data_ptr={102247680,
                              116433664,
                              1482624192,
                              1483952064,
                              1484482112,
                              1484482144},
              mutations_data_ptr=set())
FuncReadWrite(fn_name='__compiled_fn_27',
              reads={'var_29', 'var_25'},
              mutations=set(),
              compiled_fn=<function aot_module_simplified.<locals>.forward at 0x7fcb89cbf7e0>,
              input_index_to_global_var_name={0: ['var_25'], 1: ['var_29']},
              output_index_to_global_var_name={0: ['var_30']},
              outputs={'var_30'},
              aliases={'L_stack0_': {'l_stack0_'}, 'L_stack1_': {'l_stack1_'}},
              nominal_inputs=['L_stack0_', 'L_stack1_'],
              nominal_param_reads=set(),
              nominal_param_to_actual_param={},
              nominal_mutations=set(),
              reads_data_ptr=set(),
              mutations_data_ptr=set())
FuncReadWrite(fn_name='__eager_fn_31',
              reads={'var_34', 'var_2', 'var_4'},
              mutations={'var_4'},
              compiled_fn=None,
              input_index_to_global_var_name={},
              output_index_to_global_var_name={},
              outputs=set(),
              aliases={},
              nominal_inputs=[],
              nominal_param_reads=set(),
              nominal_param_to_actual_param={},
              nominal_mutations=set(),
              reads_data_ptr={102460096, 116433664, 101321344, 1484219264},
              mutations_data_ptr={102460096})
FuncReadWrite(fn_name='__compiled_fn_32',
              reads={'var_30', 'var_33', 'var_34', 'var_4'},
              mutations=set(),
              compiled_fn=<function aot_module_simplified.<locals>.forward at 0x7fcb89ec39c0>,
              input_index_to_global_var_name={0: ['var_30'], 1: ['var_33']},
              output_index_to_global_var_name={0: ['var_36']},
              outputs={'var_36'},
              aliases={'L_stack0_': {'l_stack0_'}, 'L_stack1_': {'l_stack1_'}},
              nominal_inputs=['L_stack0_', 'L_stack1_'],
              nominal_param_reads={'l__self___buf', 'l__self___weight'},
              nominal_param_to_actual_param={'l__self___buf': 'L__self___buf',
                                             'l__self___weight': 'L__self___weight'},
              nominal_mutations=set(),
              reads_data_ptr=set(),
              mutations_data_ptr=set())
FuncReadWrite(fn_name='__eager_fn_35',
              reads={'var_30', 'var_4'},
              mutations=set(),
              compiled_fn=None,
              input_index_to_global_var_name={},
              output_index_to_global_var_name={},
              outputs=set(),
              aliases={},
              nominal_inputs=[],
              nominal_param_reads=set(),
              nominal_param_to_actual_param={},
              nominal_mutations=set(),
              reads_data_ptr={102460096, 1489344192},
              mutations_data_ptr=set())
"""


with (
    torch._dynamo.config.patch(
        dynamic_shapes=False,
        capture_dynamic_output_shape_ops=False,
        capture_scalar_outputs=False,
    ),
):
    torch._dynamo.reset()
    m = TestModule()
    compiled_m = torch.compile(m, fullgraph=False, dynamic=False)
    x = torch.randn(4, 4)
    y = torch.randn(4, 4)

    # ref = m(x, y)
    actual = compiled_m(x, y)
    # assert torch.allclose(ref, actual)
