def _dict_to_str(dict_):
    ret_str = ""
    for k in dict_:
        ret_str += "{}: {:.2f}\n".format(k, dict_[k])

    return ret_str


def _output_to_stdout(str_, level):
    print(f"=========== {level} =========== \n"
          f"{str_}"
          f"============================ \n")


class Logger:
    def __init__(self):
        self.info_dict = {}
        self.debug_dict = {}
        self.warning_dict = {}
        self.error_dict = {}

    def add(self, k, v, level="info"):
        if level == "info":
            self.info_dict[k] = v
        elif level == "debug":
            self.debug_dict[k] = v
        elif level == "warning":
            self.warning_dict[k] = v
        elif level == "error":
            self.error_dict[k] = v

    def dump(self):
        if self.info_dict != {}:
            _output_to_stdout(_dict_to_str(self.info_dict), "info")
            self.info_dict = {}
        if self.debug_dict != {}:
            _output_to_stdout(_dict_to_str(self.debug_dict), "debug")
            self.debug_dict = {}
        if self.warning_dict != {}:
            _output_to_stdout(_dict_to_str(self.warning_dict), "warning")
            self.warning_dict = {}
        if self.error_dict != {}:
            _output_to_stdout(_dict_to_str(self.error_dict), "error")
            self.error_dict = {}
