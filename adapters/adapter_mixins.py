import torch

from nemo.core.classes import adapter_mixins

class AdapterModuleMixinMultispeaker(
    adapter_mixins.AdapterModuleMixin
):
    def forward_enabled_adapters(
        self, 
        input: 'torch.Tensor'
    ):
        adapter_modules = []
        enabled_adapters = self.get_enabled_adapters()
        for adapter_name in enabled_adapters:
            adapter_module = self.adapter_layer[adapter_name]
            adapter_modules.append(adapter_module)

            if hasattr(adapter_module, 'adapter_strategy'):
                strategy = (
                    adapter_module.adapter_strategy
                )  # type: 'nemo.core.classes.mixins.adapter_mixin_strategies.AbstractAdapterStrategy'
            else:
                raise AttributeError(
                    f"Adapter module `{adapter_name}` does not set the value `adapter_strategy` ! "
                    f"Please set the value of the adapter's strategy with the class "
                    f"{adapter_module.__class__.__module}.{adapter_module.__class__.__name__}."
                )

            # Call a single adapter's forward, and accept its output as the new input for the next adapter.
            
            input = self.forward_single_enabled_adapter_(
                input, adapter_module, adapter_strategy=strategy
            )
        #output = strategy(input, adapter_modules, module=self)

        return input

    def forward_single_enabled_adapter_(
        self,
        input: torch.Tensor,
        adapter_module: torch.nn.Module,
        adapter_strategy: 'nemo.core.classes.mixins.adapter_mixin_strategies.AbstractAdapterStrategy',
    ):
        output = adapter_strategy(input, adapter_module, module=self)
        return output

    def freeze_enabled_adapters(self) -> None:
        adapter_names = set([])
        for module in self.modules():  # access PT subclass method via inheritance
            if hasattr(module, 'adapter_layer') and module.is_adapter_available():
                for name, config in self.adapter_cfg.items():
                    # Skip global adapter config
                    if name == self.adapter_global_cfg_key:
                        continue

                    # Check if adapter is enabled or not
                    if self.adapter_cfg[name]['enabled'] and name in module.adapter_layer:

                        # Recursively set training mode of submodules
                        #module.adapter_layer[name].train()

                        # Recursively set grad required for submodules
                        module.adapter_layer[name].adapter_freeze()

                        # unfreeze batch norm if any in the adapter submodules
                        #for mname, module_ in module.adapter_layer[name].named_modules():
                        #    if isinstance(module_, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                        #        module_.track_running_stats = (
                        #            True  # prevent running stats from updated during finetuning
                        #        )
                                #logging.info(f"Froze adapter module {mname}: {module_}")

                        #adapter_names.add(name)

        #for name in adapter_names:
            #logging.info(f"Frozen adapter : {name}")