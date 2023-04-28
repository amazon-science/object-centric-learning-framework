# Models

A model in OCLF can be defined in one of two ways:

 1. A [Combined][ocl.utils.routing.Combined] model which is initialized using a
    dict
 2. A code model which works similar to a regular pytorch model familiar from
    other frameworks

## Combined model
Combined models are different from how models are typically written in pytorch.
The main difference is that routing of information is not performed in code,
but instead in the configuration.  This is useful if you might need access to
different inputs dependent on the exact submodule you are using.

For example, assume you want to train slot attention on images, but instead of
using the typical random initialization for slots you would like to condition
each slot on the center of mass of each object (similar to what is done in SAVi).
In this case it would be necessary to either create special handling in the
main model for it to understand which type of conditioning module you are using
(random or bbox-based conditioning) and then forward the correct inputs to the
module.  This introduces unnecessary dependencies between the code of the model
and the conditioning approach used which we would ideally like to avoid.

Additionally, models defined in code cannot be composed.  Thus if a
representation should be used for multiple prediction endpoints, these need to
be defined in code.  If new endpoints are added the code needs to be changed
and additional clauses for handling subsets of the functionality need to be introduced.

Combined models simply specify individual modules or parts of a model as
entries in a dictionary.  For instance (below uses [hydras instantiate
notation](https://hydra.cc/docs/advanced/instantiate_objects/overview/)):

```yaml title="Example model"
models:
  _target_: ocl.utils.routing.Combined

  feature_extractor:
    _target_: routed.my.feature.extractor
    my_parameter_1: test
    input_path: input.image

  grouping:
    _target_: routed.my.grouping.module
    my_parameter_2: "some other value"
    input_features_path: feature_extractor
```

Which translates into python code similar to:

```python
models = ocl.utils.routing.Combined(
  feature_extractor=routed.my_module.TestModule(
    my_parameter="test",
    input_path="input.image"
  ),
  grouping=routed.my.grouping.module(
    my_parameter_2="some other value",
    input_features_path="feature_extractor"
  )
)
```

### How does this work?
The [Combined][ocl.utils.routing.Combined] model will go through the two
modules (`feature_extractor` and `grouping`) in order, execute them and store
the return value of each module under the key same key as the module itself in
a dictionary.  This dictionary is initialized with one entry `input`, which
contains the input data and is updated with additional values as the modules
are executed.

Yet, how do the modules access the right inputs? This possible due to the magic
of the [routed][] package which automatically subclasses any module being
imported from it's path and adds routing parameters to its constructor.  In
particular, it examines the signature of the `forward`, `update` or `__call__`
method of the class that should be routed and adds
`<method_argument_name>_path` to the constructor of the class.  This allows the
routed version of the class to simply be called using dictionary that is being
expanded with each module call.  Thus each module will be able to access
outputs of modules which have been called before itself.  The path is
dot-separated and internally uses the
[get_tree_element][ocl.utils.trees.get_tree_element] implementation to derive
elements from the dict.  It thus is also possible to select elements from
nested dictionaries, lists, or even dataclasses.  Please check out the
documentation of [routed][] for more information.

### Recurrent model components
If parts of the model need to be applied individually over time the special
[Recurrent][ocl.utils.routing.Recurrent] module can be used to implement this.
It allows defining the model components that should be applied over time (or in
fact any other axis) and takes as arguments the axis along which the input
should be sliced, which input tensors should be sliced and how the initial
input should be constructed.

To access the output of the previous iteration, the entry `previous_output` can
be used.  An example of applying the [Recurrent][ocl.utils.routing.Recurrent]
can be found in
[/configs/experiment/SAVi/cater.yaml][configsexperimentsavicateryaml].

Of course, alternatively the functionality of handling higher dimensional data
can also be implemented in the module itself.


## Regular pytorch model
A model can also be defined in a similarly to regular pytorch models by
implementing (potentially only parts of) the routing in code.  One such example
is shown in [ocl.models.savi.SAVi][].  Here the model simply accepts the input
dictionary as input and internally routes information however desired.
