"""
Wrapper around logging to provide our own functionality.

This adds the ability to log using str.format() instead of %.
"""
import itertools
import logging
import os
import sys
import io
import traceback
import threading
import contextlib
from types import TracebackType
from typing import (
    Dict, Tuple, Union, Type, Callable, Any, cast, Optional,
    Generator, TextIO,
)


CTX_STACK = threading.local()


class LogMessage:
    """Allow using str.format() in logging messages.

    The __str__() method performs the joining.
    """
    def __init__(
        self,
        fmt: str,
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
    ) -> None:
        self.fmt = fmt
        self.args = args
        self.kwargs = kwargs
        self.has_args = bool(kwargs or args)

    def format_msg(self) -> str:
        """Format using str.format."""
        # Only format if we have arguments!
        # That way { or } can be used in regular messages.
        if self.has_args:
            f = self.fmt = str(self.fmt).format(*self.args, **self.kwargs)

            # Don't repeat the formatting, and don't keep refs to the args.
            del self.args, self.kwargs
            self.has_args = False
            return f
        else:
            return str(self.fmt)

    def __str__(self) -> str:
        """Format the string, and add an ASCII indent."""
        msg = self.format_msg()

        if '\n' not in msg:
            return msg

        # For multi-line messages, add an indent so they're associated
        # with the logging tag.
        lines = msg.split('\n')
        if lines[-1].isspace():
            # Strip last line if it's blank
            del lines[-1]
        # [I] A multi-line message looks like the following:
        # | blah
        # | blah
        # |___
        #
        return '\n | '.join(lines[:]) + '\n |___\n'


class LoggerAdapter(logging.LoggerAdapter):
    """Fix loggers to use str.format().

    """
    def __init__(self, logger: logging.Logger, alias: str=None) -> None:
        # Alias is a replacement module name for log messages.
        self.alias = alias
        self.logger = logger
        logging.LoggerAdapter.__init__(self, logger, extra={})

    def log(
        self,
        level: int,
        msg: str,
        *args: object,
        exc_info: Union[BaseException, Tuple[Type[BaseException], BaseException, TracebackType]]=None,
        stack_info: bool=False,
        **kwargs: object,
    ):
        """This version of .log() is for str.format() compatibility.

        The message is wrapped in a LogMessage object, which is given the
        args and kwargs.
        """
        if self.isEnabledFor(level):
            try:
                ctx = CTX_STACK.stack[-1]
            except (AttributeError, IndexError):
                ctx = ''

            self.logger._log(
                level,
                LogMessage(msg, args, kwargs),
                (),  # No positional arguments, we do the formatting through
                # LogMessage..
                # Pull these two arguments out of kwargs, so they can be set..
                exc_info=exc_info,
                stack_info=stack_info,
                extra={
                    'alias': self.alias,
                    'context': f' ({ctx})' if ctx else '',
                },
            )

    def __getattr__(self, attr: str) -> Any:
        """Delegate unknown methods to the logger."""
        return getattr(self.logger, attr)


class Formatter(logging.Formatter):
    """Override exception handling."""
    SKIP_LIBS = ['importlib', 'cx_freeze', 'PyInstaller']
    def formatException(self, ei: Union[
        Tuple[Type[BaseException], BaseException, Optional[TracebackType]],
        Tuple[None, None, None],
    ]) -> str:
        """Ignore importlib, cx_freeze and PyInstaller."""
        exc_type, exc_value, exc_tb = ei
        buffer = io.StringIO()

        trace: Optional[TracebackType] = exc_tb

        try:
            while trace is not None:
                filename = trace.tb_frame.f_code.co_filename.casefold()
                if all(keyword not in filename for keyword in self.SKIP_LIBS):
                    break
                trace = trace.tb_next

            if trace is None:
                # It's to do with these libraries themselves? Show the full
                # traceback.
                trace = exc_tb
        except Exception:  # Something failed, keep the original.
            trace = exc_tb

        if exc_type is not None and exc_value is not None and exc_tb is not None:
            for line in traceback.TracebackException(exc_type, exc_value, trace).format():
                buffer.write(line)

        return buffer.getvalue().rstrip('\n')

    def format(self, record: logging.LogRecord) -> str:
        record.__dict__.setdefault('context', '')
        return super().format(record)


def get_handler(filename: str) -> logging.FileHandler:
    """Cycle log files, then give the required file handler."""
    name, ext = os.path.splitext(filename)

    suffixes = ('.5', '.4', '.3', '.2', '.1', '')

    try:
        # Remove the oldest one.
        try:
            os.remove(name + suffixes[0] + ext)
        except FileNotFoundError:
            pass

        # Go in reverse, moving each file over to give us space.
        for frm, to in zip(suffixes[1:], suffixes):
            try:
                os.rename(name + frm + ext, name + to + ext)
            except FileNotFoundError:
                pass

        try:
            return logging.FileHandler(filename, mode='x')
        except FileExistsError:
            pass
    except PermissionError:
        pass

    # On windows, we can't touch files opened by other programs (ourselves).
    # If another copy of us is open, it'll hold access.
    # In that case, just keep trying suffixes until we find an empty file.
    for ind in itertools.count(start=1):
        try:
            return logging.FileHandler(
                '{}.{}{}'.format(name, ind, ext),
                mode='x',
            )
        except (FileExistsError, PermissionError):
            pass
    else:
        raise AssertionError  # Never terminates


class NullStream(io.IOBase, TextIO):
    """A stream object that discards all data.

    This is needed for multiprocessing, since it tries to flush stdout.
    That'll fail if it is None.
    """
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def write(*args: Any, **kwargs: Any) -> None:
        """Write nothing to the file."""
        pass

    @staticmethod
    def read(*args: Any, **kwargs: Any) -> str:
        """We never have data."""
        return ''


class NewLogRecord(logging.LogRecord):
    """Allow passing an alias and context for log modules."""
    # This breaks %-formatting, so only set when init_logging() is called.
    alias: Optional[str] = None

    def getMessage(self):
        """We have to hook here to change the value of .module.

        It's called just before the formatting call is made.
        """
        if self.alias is not None:
            self.module = self.alias
        return str(self.msg)


def init_logging(
    filename: Union[str, os.PathLike]=None,
    main_logger: str='',
    on_error: Callable[
        [Type[BaseException], BaseException, TracebackType],
        None,
    ]=None,
) -> logging.Logger:
    """Setup the logger and logging handlers.

    If filename is set, all logs will be written to this file as well.
    This also sets sys.except_hook, so uncaught exceptions are captured.
    on_error should be a function to call when uncaught exceptions are thrown.
    (taking type, value, traceback).
    If the exception is a BaseException, the app will quit silently.
    """
    if logging.getLogRecordFactory() is not logging.LogRecord:
        raise ValueError('Unknown record factory: ', logging.getLogRecordFactory())
    logging.setLogRecordFactory(NewLogRecord)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Put more info in the log file, since it's not onscreen.
    long_log_format = Formatter(
        '[{levelname}]{context} {module}.{funcName}(): {message}',
        style='{',
    )
    # Console messages, etc.
    short_log_format = Formatter(
        # One letter for level name
        '[{levelname[0]}]{context} {module}.{funcName}(): {message}',
        style='{',
    )

    if filename is not None:
        # Make the directories the logs are in, if needed.
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # The log contains DEBUG and above logs.
        log_handler = get_handler(filename)
        log_handler.setLevel(logging.DEBUG)
        log_handler.setFormatter(long_log_format)
        logger.addHandler(log_handler)

    if sys.stdout:
        stdout_loghandler = logging.StreamHandler(sys.stdout)
        stdout_loghandler.setLevel(
            logging.DEBUG
            if os.environ.get('SRCTOOLS_DEBUG', 0) == '1' else
            logging.INFO
        )
        stdout_loghandler.setFormatter(short_log_format)
        logger.addHandler(stdout_loghandler)

        if sys.stderr:
            def ignore_warnings(record: logging.LogRecord):
                """Filter out messages higher than WARNING.

                Those are handled by stdError, and we don't want duplicates.
                """
                return record.levelno < logging.WARNING
            stdout_loghandler.addFilter(ignore_warnings)
    else:
        sys.stdout = NullStream()

    if sys.stderr:
        stderr_loghandler = logging.StreamHandler(sys.stderr)
        stderr_loghandler.setLevel(logging.WARNING)
        stderr_loghandler.setFormatter(short_log_format)
        logger.addHandler(stderr_loghandler)
    else:
        sys.stderr = NullStream()

    # Use the exception hook to report uncaught exceptions, and finalise the
    # logging system.
    old_except_handler = sys.excepthook

    def except_handler(
        exc_type: Type[BaseException],
        exc_value: BaseException,
        exc_tb: TracebackType,
    ) -> None:
        """Log uncaught exceptions."""
        if not issubclass(exc_type, Exception):
            # It's subclassing BaseException (KeyboardInterrupt, SystemExit),
            # so we should quit without messages.
            return

        logger.error(
            'Uncaught Exception:',
            exc_info=(exc_type, exc_value, exc_tb),
        )
        if on_error is not None:
            on_error(exc_type, exc_value, exc_tb)
        # Call the original handler - that prints to the normal console.
        if old_except_handler is not sys.__excepthook__:
            old_except_handler(exc_type, exc_value, exc_tb)

    sys.excepthook = except_handler

    if main_logger:
        return get_logger(main_logger)
    else:
        return cast(logging.Logger, LoggerAdapter(logger))


def get_logger(name: str='', alias: str=None) -> logging.Logger:
    """Get the named logger object.

    This puts the logger into the srctools namespace, and wraps it to
    use str.format() instead of % formatting.
    If set, alias is the name to show for the module.
    """
    if name:
        log = logging.getLogger('srctools.' + name)
    else:  # Allow retrieving the main logger.
        log = logging.getLogger('srctools')
    return cast(logging.Logger, LoggerAdapter(log, alias))


@contextlib.contextmanager
def context(name: str) -> Generator[str, None, None]:
    """Allows specifying additional information for any logs contained in this block.

    The specified string gets included in the log messages.
    """
    try:
        stack = CTX_STACK.stack
    except AttributeError:
        stack = CTX_STACK.stack = []
    stack.append(name)
    try:
        yield name
    finally:
        popped = stack.pop()
        assert popped is name, f'Popped incorrect value: pop({popped!r}) != ctx({name!r})!'
