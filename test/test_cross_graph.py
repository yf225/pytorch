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

global_a = torch.randn(4, 4, device="cuda")

@disable()
def g2_read_global_var(a, b):
    return torch.cat(torch.chunk(a * b.div(torch.selu(global_a)), 2))

@torch._dynamo.disable()
def g2_read_global_var_simple(a, b):
    k = a * b.div(global_a)
    return torch.cat(torch.chunk(k, 2))

# def f(a, b):
#     return a + b + g2_read_global_var_simple(a, b)

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
        z = torch.relu(x) + g1_mutation_tuple(x, y)[0]
        z = z + g1_mutation_tensor(x, x)
        z = z + g2(x, y)
        z = x + y
        z = z + g2_read_global_var(x, y)
        z = z + self.f_read_param_mutate_param(x)
        z = z + torch.tanh(self.weight)
        z = z + self.buf
        z = z + global_a
        return z

"""
var_2: x
var_3: y
var_4: self.buf
var_5: torch.relu(x)
var_6: y
var_10: z = torch.relu(x) + g1_mutation_tuple(x, y)[0]
var_14: g1_mutation_tensor(x, x)
var_15: z = z + g1_mutation_tensor(x, x)
var_19: g2(x, y)
var_20: z = z + g2(x, y)
var_24: g2_read_global_var(x, y)
var_25: z = z + g2_read_global_var(x, y)
var_28: self.f_read_param_mutate_param(x)
var_29: global_a
var_30: self.weight

# x.relu_()
# self.buf.relu_()
# y = torch.cat(torch.chunk(y, 2))
# z = torch.relu(x)
FuncReadWrite(fn_name='__compiled_fn_0',
              reads={'var_3', 'var_4', 'var_2'},
              mutations={'var_4', 'var_2'},
              compiled_fn=<function aot_module_simplified.<locals>.forward at 0x7fab2bdca520>,
              input_index_to_global_var_name={0: ['var_2'], 1: ['var_3']},
              output_index_to_global_var_name={0: ['var_5'], 1: ['var_6']},
              outputs={'var_5', 'var_6'},
              aliases={'L_x_': {'l_x_'},
                       'L_y_': {'l_y_'},
                       'chunk': {'getitem_1', 'getitem'},
                       'l__self___buf': {'relu__1'},
                       'l_x_': {'relu_'},
                       'l_y_': {'chunk'}},
              nominal_inputs=['L_x_', 'L_y_'],
              nominal_outputs=['relu', 'cat_1'],
              nominal_param_reads={'l__self___buf'},
              nominal_param_to_actual_param={'l__self___buf': 'L__self___buf'},
              nominal_mutations={'l_x_', 'l__self___buf'},
              reads_data_ptr=set(),
              mutations_data_ptr=set())

# g1_mutation_tuple(x, y)[0]
FuncReadWrite(fn_name='__eager_fn_7',
              reads={'var_2'},
              mutations={'var_2'},
              compiled_fn=None,
              input_index_to_global_var_name={},
              output_index_to_global_var_name={},
              outputs=set(),
              aliases={},
              nominal_inputs=[],
              nominal_outputs=[],
              nominal_param_reads=set(),
              nominal_param_to_actual_param={},
              nominal_mutations=set(),
              reads_data_ptr={'140368259581952_1477853872'},
              mutations_data_ptr={'140368259581952_1477853872'})
FuncReadWrite(fn_name='__compiled_fn_8',
              reads={'var_5', 'var_2'},
              mutations=set(),
              compiled_fn=<function aot_module_simplified.<locals>.forward at 0x7fab2baed1c0>,
              input_index_to_global_var_name={0: ['var_5'], 1: ['var_2']},
              output_index_to_global_var_name={0: ['var_10']},
              outputs={'var_10'},
              aliases={'L_stack0_': {'l_stack0_'},
                       'L_stack1_0_': {'l_stack1_0_'}},
              nominal_inputs=['L_stack0_', 'L_stack1_0_'],
              nominal_outputs=['z'],
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
              nominal_outputs=[],
              nominal_param_reads=set(),
              nominal_param_to_actual_param={},
              nominal_mutations=set(),
              reads_data_ptr={'140368259581952_1477853872'},
              mutations_data_ptr={'140368259581952_1477853872'})
FuncReadWrite(fn_name='__compiled_fn_12',
              reads={'var_14', 'var_10'},
              mutations=set(),
              compiled_fn=<function aot_module_simplified.<locals>.forward at 0x7fab2b5cb9c0>,
              input_index_to_global_var_name={0: ['var_10'], 1: ['var_14']},
              output_index_to_global_var_name={0: ['var_15']},
              outputs={'var_15'},
              aliases={'L_stack0_': {'l_stack0_'}, 'L_stack1_': {'l_stack1_'}},
              nominal_inputs=['L_stack0_', 'L_stack1_'],
              nominal_outputs=['z'],
              nominal_param_reads=set(),
              nominal_param_to_actual_param={},
              nominal_mutations=set(),
              reads_data_ptr=set(),
              mutations_data_ptr=set())
FuncReadWrite(fn_name='__eager_fn_16',
              reads={'var_6', 'var_2'},
              mutations=set(),
              compiled_fn=None,
              input_index_to_global_var_name={},
              output_index_to_global_var_name={},
              outputs=set(),
              aliases={},
              nominal_inputs=[],
              nominal_outputs=[],
              nominal_param_reads=set(),
              nominal_param_to_actual_param={},
              nominal_mutations=set(),
              reads_data_ptr={'140368259581952_1477853872',
                              '140368259582976_1479154704',
                              '140368259585536_1480743776'},
              mutations_data_ptr=set())
FuncReadWrite(fn_name='__compiled_fn_17',
              reads={'var_2', 'var_6', 'var_19', 'var_15'},
              mutations=set(),
              compiled_fn=<function aot_module_simplified.<locals>.forward at 0x7fab2bcaf920>,
              input_index_to_global_var_name={0: ['var_15'],
                                              1: ['var_19'],
                                              2: ['var_2'],
                                              3: ['var_6']},
              output_index_to_global_var_name={0: ['var_20']},
              outputs={'var_20'},
              aliases={'L_stack0_': {'l_stack0_'},
                       'L_stack1_': {'l_stack1_'},
                       'L_x_': {'l_x_'},
                       'L_y_': {'l_y_'}},
              nominal_inputs=['L_stack0_', 'L_stack1_', 'L_x_', 'L_y_'],
              nominal_outputs=['z_1'],
              nominal_param_reads=set(),
              nominal_param_to_actual_param={},
              nominal_mutations=set(),
              reads_data_ptr=set(),
              mutations_data_ptr=set())
FuncReadWrite(fn_name='__eager_fn_21',
              reads={'var_6', 'var_29', 'var_2', 'var_24'},
              mutations=set(),
              compiled_fn=None,
              input_index_to_global_var_name={},
              output_index_to_global_var_name={},
              outputs=set(),
              aliases={},
              nominal_inputs=[],
              nominal_outputs=[],
              nominal_param_reads=set(),
              nominal_param_to_actual_param={},
              nominal_mutations=set(),
              reads_data_ptr={'140368259579904_107522320',
                              '140368259581952_1477853872',
                              '140368259582976_1479154704',
                              '140368259586560_1481007200',
                              '140368259587072_1481201472'},
              mutations_data_ptr=set())
FuncReadWrite(fn_name='__compiled_fn_22',
              reads={'var_20', 'var_24'},
              mutations=set(),
              compiled_fn=<function aot_module_simplified.<locals>.forward at 0x7fab2ba72e80>,
              input_index_to_global_var_name={0: ['var_20'], 1: ['var_24']},
              output_index_to_global_var_name={0: ['var_25']},
              outputs={'var_25'},
              aliases={'L_stack0_': {'l_stack0_'}, 'L_stack1_': {'l_stack1_'}},
              nominal_inputs=['L_stack0_', 'L_stack1_'],
              nominal_outputs=['z'],
              nominal_param_reads=set(),
              nominal_param_to_actual_param={},
              nominal_mutations=set(),
              reads_data_ptr=set(),
              mutations_data_ptr=set())
FuncReadWrite(fn_name='__eager_fn_26',
              reads={'var_30', 'var_4', 'var_2'},
              mutations={'var_4'},
              compiled_fn=None,
              input_index_to_global_var_name={},
              output_index_to_global_var_name={},
              outputs=set(),
              aliases={},
              nominal_inputs=[],
              nominal_outputs=[],
              nominal_param_reads=set(),
              nominal_param_to_actual_param={},
              nominal_mutations=set(),
              reads_data_ptr={'140368259580928_1380803600',
                              '140368259581440_1381212240',
                              '140368259581952_1477853872',
                              '140368259588096_97054160'},
              mutations_data_ptr={'140368259581440_1381212240'})
FuncReadWrite(fn_name='__compiled_fn_27',
              reads={'var_30', 'var_25', 'var_4', 'var_29', 'var_28'},
              mutations=set(),
              compiled_fn=<function aot_module_simplified.<locals>.forward at 0x7fab2ba12840>,
              input_index_to_global_var_name={0: ['var_25'],
                                              1: ['var_28'],
                                              2: ['var_29']},
              output_index_to_global_var_name={0: ['var_32']},
              outputs={'var_32'},
              aliases={'G_global_a_': {'g_global_a_'},
                       'L_stack0_': {'l_stack0_'},
                       'L_stack1_': {'l_stack1_'}},
              nominal_inputs=['L_stack0_', 'L_stack1_', 'G_global_a_'],
              nominal_outputs=['z_3'],
              nominal_param_reads={'l__self___buf', 'l__self___weight'},
              nominal_param_to_actual_param={'l__self___buf': 'L__self___buf',
                                             'l__self___weight': 'L__self___weight'},
              nominal_mutations=set(),
              reads_data_ptr=set(),
              mutations_data_ptr=set())
FuncReadWrite(fn_name='__eager_fn_31',
              reads={'var_25', 'var_4', 'var_29'},
              mutations=set(),
              compiled_fn=None,
              input_index_to_global_var_name={},
              output_index_to_global_var_name={},
              outputs=set(),
              aliases={},
              nominal_inputs=[],
              nominal_outputs=[],
              nominal_param_reads=set(),
              nominal_param_to_actual_param={},
              nominal_mutations=set(),
              reads_data_ptr={'140368259579904_107522320',
                              '140368259581440_1381212240',
                              '140368259586560_1479963312'},
              mutations_data_ptr=set())
"""

"""
>>> x = torch.randn(4, 4, device="cuda")
>>> x.stride()
(4, 1)
>>> x.data_ptr()
140626427379712
>>> x.storage().data_ptr()
<stdin>:1: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
140626427379712
>>> x.storage()._cdata
100713264
>>> x._cdata
1469156160
>>> y = x[1]
>>> y.data_ptr()
140626427379728
>>> y.storage().data_ptr()
140626427379712
>>> y.storage()._cdata
100713264
>>> y._cdata
115315808
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
    m = m.cuda()
    compiled_m = torch.compile(m, fullgraph=False, dynamic=False)
    x = torch.randn(4, 4, device="cuda")
    y = torch.randn(4, 4, device="cuda")

    # ref = m(x, y)
    actual = compiled_m(x, y)
    # assert torch.allclose(ref, actual)
