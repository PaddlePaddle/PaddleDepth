# -*- coding: UTF-8 -*-
from SysBasic import ShowHandler
from SysBasic import LogHandler as log
from ._meta_mode import MetaMode


class TestProc(MetaMode):
    def __init__(self, args: object, user_inference_func: object,
                 is_training: bool = True) -> None:
        super().__init__(args, user_inference_func, is_training)
        self.__args = args

    def __init_testing_para(self, epoch: int, is_training: bool = True) -> tuple:
        total_iteration, off_set = 0, 1
        self._graph.set_model_mode(is_training)
        dataloader = self._data_manager.get_dataloader(is_training)
        self._graph.set_model_mode(False)
        self._graph.user_pretreatment(epoch)
        return total_iteration, off_set, dataloader

    def _testing_data_proc(self, batch_data: list) -> tuple:
        graph, data_manager = self._get_graph_and_data_manager
        input_data, supplement = data_manager.user_split_data(batch_data, False)
        outputs_data = graph.exec(input_data, None)
        return outputs_data, supplement

    @ShowHandler.show_method
    def _show_testing_proc(self, total_iteration: int) -> None:
        self.calculate_ave_runtime(total_iteration, self._training_iteration)
        self.update_show_bar('')

    @ShowHandler.show_method
    def _testing_post_proc(self) -> None:
        self.stop_show_setting()
        log.info("Finish testing process!")

    def __test_loop(self) -> None:
        total_iteration, off_set, dataloader = self.__init_testing_para(0, True)
        self.init_show_setting(self._training_iteration, "Test")
        log.info("Start testing iteration!")
        for iteration, batch_data in enumerate(dataloader):
            total_iteration = iteration + off_set
            outputs_data, supplement = self._testing_data_proc(batch_data)
            self._save_result(iteration, outputs_data, supplement)
            self._show_testing_proc(total_iteration)
        self._graph.user_post_process(0)

    def __preparation_proc(self) -> None:
        self._graph.restore_model()

    def exec(self) -> None:
        self._init_data_model_handler()
        log.info("Start the testing process!")
        self.__preparation_proc()
        self.__test_loop()
        self._testing_post_proc()
