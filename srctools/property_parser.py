"""Reads and parses Valve's KeyValues files.

These files follow the following general format:
    "Name"
        {
        "Name" "Value" // Comment
        "Name"
            {
            "Name" "Value"
            }
        "Name" "Value"
        "SecondName" "ps3-only value" [ps3]
        "Third_Name" "never on Linux" [!linux]
        "Name" "multi-line values
are supported like this.
They end with a quote."
        }

    The names are usually case-insensitive.
    Call 'Property(name, value) to get a property object, or
    Property.parse(file, 'name') to parse a file.

    This will perform a round-trip file read:
    >>> with open('filename.txt', 'r') as f:
    ...     props = Property.parse(f, 'filename.txt')
    >>> with open('filename_2.txt', 'w') as f:
    ...     for line in props.export():
    ...         f.write(line)

    Property values should be either a string, or a list of children Properties.
    Names will be converted to lowercase automatically; use Prop.real_name to
    obtain the original spelling. To allow multiple root blocks in a file, the
    returned property from Property.parse() has a name of None - this property
    type will export with un-indented children.

    Properties with children can be indexed by their names, or by a
    ('name', default) tuple:

    >>> props = Property('Top', [
    ...     Property('child1', '1'),
    ...     Property('child2', '0'),
    ... ])
    ... props['child1']
    '1'
    >>> props['child3']
    Traceback (most recent call last):
        ...
    IndexError: No key child3!
    >>> props['child3', 'default']
    'default'
    >>> props['child4', object()]
    <object object ax 0x...>
    >>> del props['child2']
    >>> props['child3'] = 'new value'
    >>> props
    Property('Top', [Property('child1', '1'), Property('child3', 'new value')])

    \n, \t, and \\ will be converted in Property values.
"""
import sys
import keyword
import builtins  # Property.bool etc shadows these.
import warnings

from srctools import BOOL_LOOKUP, EmptyMapping
from srctools.math import Vec as _Vec
from srctools.tokenizer import BaseTokenizer, Token, Tokenizer, TokenSyntaxError, escape_text

from typing import (
    Optional, Union, Any,
    List, Tuple, Dict, Iterator,
    TypeVar,
    Iterable,
    overload,
    Callable,
    Mapping,
)


__all__ = ['KeyValError', 'NoKeyError', 'Property']

# Sentinel value to indicate that no default was given to find_key()
_NO_KEY_FOUND = object()

_Prop_Value = Union[List['Property'], str, Any]
_As_Dict_Ret = Dict[str, Union[str, '_As_Dict_Ret']]

T = TypeVar('T')

# Various [flags] used after property names in some Valve files.
# See https://github.com/ValveSoftware/source-sdk-2013/blob/master/sp/src/tier1/KeyValues.cpp#L2055
PROP_FLAGS_DEFAULT = {
    # We know we're not on a console...
    'x360': False,
    'ps3': False,
    'gameconsole': False,

    'win32': True,  # Not actually windows, it actually means 'PC'
    'osx': sys.platform.startswith('darwin'),
    'linux': sys.platform.startswith('linux'),
}


class KeyValError(TokenSyntaxError):
    """An error that occurred when parsing a Valve KeyValues file.

    mess = The error message that occurred.
    file = The filename passed to Property.parse(), if it exists
    line_num = The line where the error occurred.
    """

    
class NoKeyError(LookupError):
    """Raised if a key is not found when searching from find_key().

    key = The missing key that was asked for.
    """
    def __init__(self, key: str) -> None:
        super().__init__()
        self.key = key

    def __repr__(self) -> str:
        return 'NoKeyError({!r})'.format(self.key)

    def __str__(self) -> str:
        return "No key " + self.key + "!"


def _read_flag(flags: Mapping[str, bool], flag_val: str) -> bool:
    """Check whether a flag is True or False."""
    flag_inv = flag_val[:1] == '!'
    if flag_inv:
        flag_val = flag_val[1:]
    flag_val = flag_val.casefold()
    try:
        flag_result = bool(flags[flag_val])
    except KeyError:
        flag_result = PROP_FLAGS_DEFAULT.get(flag_val, False)
    # If flag succeeds
    return flag_inv is not flag_result


class Property:
    """Represents Property found in property files, like those used by Valve.

    Value should be a string (for leaf properties), or a list of children
    Property objects.
    The name should be a string, or None for a root object.
    Root objects export each child at the topmost indent level.
        This is produced from Property.parse() calls.
    """
    # Helps decrease memory footprint with lots of Property values.
    __slots__ = ('_folded_name', 'real_name', '_value')
    _folded_name: Optional[str]
    real_name: Optional[str]
    _value: _Prop_Value

    def __init__(
        self: 'Property',
        name: Optional[str],
        value: _Prop_Value,
    ) -> None:
        """Create a new property instance.

        """
        if name is None:
            self._folded_name = self.real_name = None
        else:
            self.real_name = sys.intern(name)
            self._folded_name = sys.intern(name.casefold())

        self._value = value

    @property
    def value(self) -> _Prop_Value:
        """Return the value of a leaf property."""
        if self.has_children():
            warnings.warn("Accessing internal property block list is deprecated", DeprecationWarning, 2)
        return self._value

    @value.setter
    def value(self, value: str) -> None:
        """Set the value of a leaf property."""
        if not isinstance(value, str):
            warnings.warn("Setting properties to non-string values will be fixed soon.", DeprecationWarning, 2)
        self._value = value

    @property
    def name(self) -> Optional[str]:
        """Name automatically casefolds() any given names.

        This ensures comparisons are always case-insensitive.
        Read .real_name to get the original value.
        """
        return self._folded_name

    @name.setter
    def name(self, new_name: Optional[str]) -> None:
        if new_name is None:
            self._folded_name = self.real_name = None
        else:
            # Intern names to help reduce duplicates in memory.
            self.real_name = sys.intern(new_name)
            self._folded_name = sys.intern(new_name.casefold())

    def edit(self, name: Optional[str]=None, value: Optional[str]=None) -> 'Property':
        """Simultaneously modify the name and value."""
        if name is not None:
            self.real_name = name
            self._folded_name = name.casefold()
        if value is not None:
            self.value = value
        return self

    @staticmethod
    def parse(
        file_contents: Union[str, BaseTokenizer, Iterator[str]],
        filename='', *,
        flags: Mapping[str, bool]=EmptyMapping,
        allow_escapes: bool=True,
        single_line: bool=False,
    ) -> "Property":
        """Returns a Property tree parsed from given text.

        filename, if set should be the source of the text for debug purposes.
        file_contents should be an iterable of strings or a single string.
        flags should be a mapping for additional flags to accept
        (which overrides defaults).
        allow_escapes allows choosing if \\t or similar escapes are parsed.
        If single_line is set, allow multiple properties to be on the same line.
        This means unterminated strings will be caught late (if at all), but
        it allows parsing some 'internal' data blocks.

        Alternatively, file_contents may be an already created tokenizer.
        In this case allow_escapes is ignored.
        """
        # The block we are currently adding to.

        # The special name 'None' marks it as the root property, which
        # just outputs its children when exported. This way we can handle
        # multiple root blocks in the file, while still returning a single
        # Property object which has all the methods.
        # Skip calling __init__ for speed.
        cur_block = root = Property.__new__(Property)
        cur_block._folded_name = cur_block.real_name = None
        cur_block._value = []

        # A queue of the properties we are currently in (outside to inside).
        # And the line numbers of each of these, for error reporting.
        open_properties = [(cur_block, 1)]

        # Grab a reference to the token values, so we avoid global lookups.
        STRING = Token.STRING
        PROP_FLAG = Token.PROP_FLAG
        NEWLINE = Token.NEWLINE
        BRACE_OPEN = Token.BRACE_OPEN
        BRACE_CLOSE = Token.BRACE_CLOSE

        if isinstance(file_contents, BaseTokenizer):
            tokenizer = file_contents
            tokenizer.filename = filename
            tokenizer.error_type = KeyValError
        else:
            tokenizer = Tokenizer(
                file_contents,
                filename,
                KeyValError,
                string_bracket=True,
                allow_escapes=allow_escapes,
            )

        # If >= 0, we're requiring a block to open next ("name"\n must have { next.)
        # It's the line number of the header name then.
        # If -1, there's no block.
        # If -2, the block is disabled, so we need to skip it.
        block_line = -1
        # Are we permitted to replace the last property with a flagged version of the same?
        can_flag_replace = False

        for token_type, token_value in tokenizer:
            if token_type is BRACE_OPEN:  # {
                # Open a new block - make sure the last token was a name..
                if block_line == -1:
                    raise tokenizer.error(
                        'Property cannot have sub-section if it already '
                        'has an in-line value.\n\n'
                        'A "name" "value" line cannot then open a block.',
                    )
                can_flag_replace = False
                if block_line == -2:
                    # It failed the flag check, use a dummy property
                    # so it's just discarded.
                    cur_block = Property.__new__(Property)
                    cur_block._folded_name = cur_block.real_name = '<skipped>'
                else:
                    cur_block = cur_block._value[-1]
                cur_block._value = []
                open_properties.append((cur_block, block_line))
                block_line = -1
                continue
            # Something else, but followed by '{'
            elif block_line != -1 and token_type is not NEWLINE:
                raise tokenizer.error(
                    'Block opening ("{{") required!\n\n'
                    'A single "name" on a line should next have a open brace '
                    'to begin a block.',
                )

            if token_type is NEWLINE:
                continue
            if token_type is STRING:   # "string"
                # Skip calling __init__ for speed. Value needs to be set
                # before using this, since it's unset here.
                keyvalue = Property.__new__(Property)
                keyvalue._folded_name = sys.intern(token_value.casefold())
                keyvalue.real_name = sys.intern(token_value)

                # We need to check the next token to figure out what kind of
                # prop it is.
                prop_type, prop_value = tokenizer()

                # It's a block followed by flag. ("name" [stuff])
                if prop_type is PROP_FLAG: 
                    # That must be the end of the line..
                    tokenizer.expect(NEWLINE)
                    if _read_flag(flags, prop_value):
                        block_line = tokenizer.line_num
                        keyvalue._value = []

                        # Special function - if the last prop was a
                        # keyvalue with this name, replace it instead.
                        if (
                            can_flag_replace and
                            cur_block._value[-1].real_name == token_value and
                            cur_block._value[-1].has_children()
                        ):
                            cur_block._value[-1] = keyvalue
                        else:
                            cur_block._value.append(keyvalue)
                        # Can't do twice in a row
                        can_flag_replace = False
                    else:
                        # Signal that the block needs to be discarded.
                        block_line = -2

                elif prop_type is STRING:
                    # A value.. ("name" "value")
                    if block_line != -1:
                        raise tokenizer.error(
                            'Keyvalue split across lines!\n\n'
                            'A value like "name" "value" must be on the same '
                            'line.'
                        )

                    keyvalue._value = prop_value

                    # Check for flags.
                    flag_token, flag_val = tokenizer()
                    if flag_token is PROP_FLAG:
                        # Should be the end of the line here.
                        tokenizer.expect(NEWLINE)
                        if _read_flag(flags, flag_val):
                            # Special function - if the last prop was a
                            # keyvalue with this name, replace it instead.
                            if (
                                can_flag_replace and
                                cur_block._value[-1].real_name == token_value and
                                type(cur_block._value[-1].value) == str
                            ):
                                cur_block._value[-1] = keyvalue
                            else:
                                cur_block._value.append(keyvalue)
                            # Can't do twice in a row
                            can_flag_replace = False
                    elif flag_token is STRING:
                        # Specifically disallow multiple text on the same line
                        # normally.
                        # ("name" "value" "name2" "value2")
                        if single_line:
                            cur_block._value.append(keyvalue)
                            tokenizer.push_back(flag_token, flag_val)
                            continue
                        else:
                            raise tokenizer.error(
                                "Cannot have multiple names on the same line!"
                            )
                    else:
                        # Otherwise, it's got nothing after.
                        # So insert the keyvalue, and check the token
                        # in the next loop. This allows braces to be
                        # on the same line.
                        cur_block._value.append(keyvalue)
                        can_flag_replace = True
                        tokenizer.push_back(flag_token, flag_val)
                    continue
                else:
                    # Something else - treat this as a block, and
                    # then re-evaluate the token in the next loop.
                    keyvalue._value = []

                    block_line = tokenizer.line_num
                    can_flag_replace = False
                    cur_block._value.append(keyvalue)
                    tokenizer.push_back(prop_type, prop_value)
                    continue

            elif token_type is BRACE_CLOSE:  # }
                # Move back a block
                open_properties.pop()
                try:
                    cur_block, _ = open_properties[-1]
                except IndexError:
                    # It's empty, we've closed one too many properties.
                    raise tokenizer.error(
                        'Too many closing brackets.\n\n'
                        'An extra closing bracket was added which would '
                        'close the outermost level.',
                    )
                # For replacing the block.
                can_flag_replace = True
            else:
                raise tokenizer.error(token_type, token_value)

        # We're out of data, do some final sanity checks.
        
        # We last had a ("name"\n), so we were expecting a block
        # next.
        if block_line != -1:
            raise KeyValError(
                'Block opening ("{") required, but hit EOF!\n'
                'A "name" line was located at the end of the file, which needs'
                ' a {} block to follow.',
                tokenizer.filename,
                line=None,
            )
        
        # All the properties in the file should be closed,
        # so the only thing in open_properties should be the 
        # root one we added.
        
        if len(open_properties) > 1:
            raise KeyValError(
                'End of text reached with remaining open sections.\n\n'
                "File ended with at least one property that didn't "
                'have an ending "}".\n'
                'Open properties: \n- Root at line 1\n' + '\n'.join([
                    f'- "{prop.real_name}" on line {line_num}'
                    for prop, line_num in open_properties[1:]
                ]),
                tokenizer.filename,
                line=None,
            )
        # Return that root property.
        return root

    def find_all(self, *keys) -> Iterator['Property']:
        """Search through the tree, yielding all properties that match a particular path.

        """
        depth = len(keys)
        if depth == 0:
            raise ValueError("Cannot find_all without commands!")

        targ_key = keys[0].casefold()
        for prop in self:
            if not isinstance(prop, Property):
                raise ValueError(
                    'Cannot find_all on a value that is not a Property!'
                )
            if prop._folded_name == targ_key is not None:
                if depth > 1:
                    if prop.has_children():
                        yield from Property.find_all(prop, *keys[1:])
                else:
                    yield prop

    def find_children(self, *keys) -> Iterator['Property']:
        """Search through the tree, yielding children of properties in a path.

        """
        for block in self.find_all(*keys):
            yield from block

    def find_key(self, key, def_: _Prop_Value=_NO_KEY_FOUND) -> 'Property':
        """Obtain the child Property with a given name.

        - If no child is found with the given name, this will return the
          default value wrapped in a Property, or raise NoKeyError if
          none is provided.
        - This prefers keys located closer to the end of the value list.
        """
        if not self.has_children():
            raise ValueError("{!r} has no children!".format(self))
        key = key.casefold()
        for prop in reversed(self._value):  # type: Property
            if prop._folded_name == key:
                return prop
        if def_ is _NO_KEY_FOUND:
            raise NoKeyError(key)
        else:
            return Property(key, def_)
            # We were given a default, return it wrapped in a Property.

    def _get_value(self, key: str, def_: T=_NO_KEY_FOUND) -> Union[_Prop_Value, T]:
        """Obtain the value of the child Property with a given name.

        Effectively find_key() but doesn't make a new property.

        - If no child is found with the given name, this will return the
          default value, or raise NoKeyError if none is provided.
        - This prefers keys located closer to the end of the value list.
        """
        if not self.has_children():
            raise ValueError("{!r} has no children!".format(self))
        key = key.casefold()
        for prop in reversed(self._value):  # type: Property
            if prop._folded_name == key:
                return prop.value
        if def_ is _NO_KEY_FOUND:
            raise NoKeyError(key)
        else:
            return def_

    def int(self, key: str, def_: T=0) -> Union[builtins.int, T]:
        """Return the value of an integer key.

        Equivalent to int(prop[key]), but with a default value if missing or
        invalid.
        If multiple keys with the same name are present, this will use the
        last only.
        """
        if not self.has_children():
            raise ValueError("{!r} has no children!".format(self))
        try:
            return int(self._get_value(key))
        except (NoKeyError, ValueError, TypeError):
            return def_

    @overload
    def float(self, key: str) -> builtins.float: ...
    @overload
    def float(self, key: str, def_: T) -> Union[builtins.float, T]: ...

    def float(self, key: str, def_: T=0.0) -> Union[builtins.float, T]:
        """Return the value of an integer key.

        Equivalent to float(prop[key]), but with a default value if missing or
        invalid.
        If multiple keys with the same name are present, this will use the
        last only.
        """
        if not self.has_children():
            raise ValueError("{!r} has no children!".format(self))
        try:
            return float(self._get_value(key))
        except (NoKeyError, ValueError, TypeError):
            return def_

    @overload
    def bool(self, key: str) -> builtins.bool: ...
    @overload
    def bool(self, key: str, def_: T) -> Union[builtins.bool, T]: ...

    def bool(self, key: str, def_: T=False) -> Union[builtins.bool, T]:
        """Return the value of an boolean key.

        The value may be case-insensitively 'true', 'false', '1', '0', 'T',
        'F', 'y', 'n', 'yes', or 'no'.
        If multiple keys with the same name are present, this will use the
        last only.
        """
        if not self.has_children():
            raise ValueError("{!r} has no children!".format(self))
        try:
            return BOOL_LOOKUP[self._get_value(key).casefold()]
        except LookupError:  # base for NoKeyError and KeyError
            return def_

    def vec(
        self, key: str,
        x: builtins.float=0.0,
        y: builtins.float=0.0,
        z: builtins.float=0.0,
    ) -> _Vec:
        """Return the given property, converted to a vector.

        If multiple keys with the same name are present, this will use the
        last only.
        """
        if not self.has_children():
            raise ValueError("{!r} has no children!".format(self))
        try:
            return _Vec.from_str(self._get_value(key), x, y, z)
        except LookupError:  # key not present, defaults.
            return _Vec(x, y, z)

    def set_key(self, path: Union[Tuple[str, ...], str], value: _Prop_Value) -> None:
        """Set the value of a key deep in the tree hierarchy.

        -If any of the hierarchy do not exist (or do not have children),
          blank properties will be added automatically
        - path should be a tuple of names, or a single string.
        """
        if not self.has_children():
            raise ValueError("{!r} has no children!".format(self))

        current_prop = self
        if isinstance(path, tuple):
            # Search through each item in the tree!
            for key in path[:-1]:
                folded_key = key.casefold()
                # We can't use find_key() here because we also
                # need to check that the property has children to search
                # through
                for prop in reversed(self._value):
                    if (prop.name is not None and
                            prop.name == folded_key and
                            prop.has_children()):
                        current_prop = prop
                        break
                else:
                    # No matching property found
                    new_prop = Property(key, [])
                    current_prop.append(new_prop)
                    current_prop = new_prop
            path = path[-1]
        try:
            current_prop.find_key(path).value = value
        except NoKeyError:
            current_prop._value.append(Property(path, value))

    def copy(self) -> 'Property':
        """Deep copy this Property tree and return it."""
        if self.has_children():
            # This recurses if needed
            return Property(
                self.real_name,
                [
                    child.copy()
                    for child in
                    self
                ]
            )
        else:
            return Property(self.real_name, self._value)

    def as_dict(self) -> _As_Dict_Ret:
        """Convert this property tree into a tree of dictionaries.

        This keeps only the last if multiple items have the same name.
        """
        if self.has_children():
            return {item._folded_name: item.as_dict() for item in self}
        else:
            return self._value

    @overload
    def as_array(self) -> List[str]: ...

    @overload
    def as_array(self, *, conv: Callable[[str], T]) -> List[T]: ...

    def as_array(self, *, conv: Callable[[str], T]=str) -> List[T]:
        """Convert a property block into a list of values.

        If the property is a single keyvalue, the single value will be
        yielded. Otherwise, each child must be a single value and each
        of those will be yielded. The name is ignored.
        """
        if self.has_children():
            arr = []
            for child in self._value:
                if child.has_children():
                    raise ValueError(
                        'Cannot have sub-children in a '
                        '"{}" array of values!'.format(self.real_name)
                    )
                arr.append(conv(child._value))
            return arr
        else:
            return [conv(self._value)]

    def __eq__(self, other: Any) -> builtins.bool:
        """Compare two items and determine if they are equal.

        This ignores names.
        """
        if isinstance(other, Property):
            return self._value == other._value
        else:
            return self._value == other  # Just compare values

    def __ne__(self, other: Any) -> builtins.bool:
        """Not-Equal To comparison. This ignores names.
        """
        if isinstance(other, Property):
            return self._value != other._value
        else:
            return self._value != other  # Just compare values

    def __len__(self) -> builtins.int:
        """Determine the number of child properties."""
        if self.has_children():
            return len(self._value)
        raise ValueError("{!r} has no children!".format(self))

    def __bool__(self) -> builtins.bool:
        """Properties are true if we have children, or have a value."""
        if self.has_children():
            return len(self._value) > 0
        else:
            return bool(self._value)

    def __iter__(self) -> Iterator['Property']:
        """Iterate through the value list.

        """
        if self.has_children():
            return iter(self._value)
        else:
            raise ValueError(
                "Can't iterate through {!r} without children!".format(self)
            )

    def iter_tree(self, blocks: builtins.bool=False) -> Iterator['Property']:
        """Iterate through all properties in this tree.

        This goes through properties in the same order that they will serialise
        into.
        If blocks is True, the property blocks will be returned as well as
        keyvalues. If false, only keyvalues will be yielded.
        """
        if self.has_children():
            return self._iter_tree(blocks)
        else:
            raise ValueError(
                "Can't iterate through {!r} without children!".format(self)
            )

    def _iter_tree(self, blocks: builtins.bool) -> Iterator['Property']:
        """Implementation of iter_tree(). This assumes self has children."""
        for prop in self._value:  # type: Property
            if prop.has_children():
                if blocks:
                    yield prop
                yield from prop._iter_tree(blocks)
            else:
                yield prop

    def __contains__(self, key: str) -> builtins.bool:
        """Check to see if a name is present in the children."""
        key = key.casefold()
        if self.has_children():
            for prop in self._value:
                if prop._folded_name == key:
                    return True
            return False

        raise ValueError("Can't search through properties without children!")

    def __getitem__(
            self,
            index: Union[
                str,
                builtins.int,
                slice,
                Tuple[Union[str, builtins.int, slice], Union[str, Any]]
            ],
            ) -> str:
        """Allow indexing the children directly.

        - If given an index, it will search by position.
        - If given a string, it will find the last Property with that name.
          (Default can be chosen by passing a 2-tuple like Prop[key, default])
        - If none are found, it raises IndexError.
        """
        if self.has_children():
            if isinstance(index, int):
                return self._value[index]
            else:
                if isinstance(index, tuple):
                    # With default value
                    return self._get_value(index[0], def_=index[1])
                else:
                    try:
                        return self._get_value(index)
                    except NoKeyError as no_key:
                        raise IndexError(no_key) from no_key
        else:
            raise ValueError("Can't index a Property without children!")

    def __setitem__(
            self,
            index: Union[builtins.int, slice, str],
            value: _Prop_Value
            ):
        """Allow setting the values of the children directly.

        If the value is a Property, this will be inserted under the given
        name or index.
        - If given an index, it will search by position.
        - If given a string, it will set the last Property with that name.
        - If none are found, it appends the value to the tree.
        - If given a tuple of strings, it will search through that path,
          and set the value of the last matching Property.
        """
        if self.has_children():
            if isinstance(index, int):
                self._value[index] = value
            else:
                if isinstance(value, Property):
                    # We don't want to assign properties, we want to add them under
                    # this name!
                    value.name = index
                    try:
                        # Replace at the same location..
                        index = self._value.index(self.find_key(index))
                    except NoKeyError:
                        self._value.append(value)
                    else:
                        self._value[index] = value
                else:
                    try:
                        self.find_key(index)._value = value
                    except NoKeyError:
                        self._value.append(Property(index, value))
        else:
            raise ValueError("Can't index a Property without children!")

    def __delitem__(self, index: Union[builtins.int, str]) -> None:
        """Delete the given property index.

        - If given an integer, it will delete by position.
        - If given a string, it will delete the last Property with that name.
        """
        if self.has_children():
            if isinstance(index, int):
                del self._value[index]
            else:
                try:
                    self._value.remove(self.find_key(index))
                except NoKeyError as no_key:
                    raise IndexError(no_key) from no_key
        else:
            raise IndexError("Can't index a Property without children!")

    def clear(self) -> None:
        """Delete the contents of a block."""
        if self.has_children():
            self._value.clear()
        else:
            raise ValueError("Can't clear a Property without children!")

    def __add__(self, other: Union[Iterable['Property'], 'Property']):
        """Allow appending other properties to this one.

        This deep-copies the Property tree first.
        Works with either a sequence of Properties or a single Property.
        """
        if self.has_children():
            copy = self.copy()
            if isinstance(other, Property):
                if other._folded_name is None:
                    copy._value.extend(other._value)
                else:
                    # We want to add the other property tree to our
                    # own, not its values.
                    copy._value.append(other)
            else:  # Assume a sequence.
                copy._value += other  # Add the values to ours.
            return copy
        else:
            return NotImplemented

    def __iadd__(self, other: Union[Iterable['Property'], 'Property']):
        """Allow appending other properties to this one.

        This is the += op, where it does not copy the object.
        """
        if self.has_children():
            if isinstance(other, Property):
                if other._folded_name is None:
                    self._value.extend(other._value)
                else:
                    self._value.append(other)
            else:
                self._value += other
            return self
        else:
            return NotImplemented

    append = __iadd__

    def merge_children(self, *names: str) -> None:
        """Merge together any children of ours with the given names.

        After execution, this tree will have only one sub-Property for
        each of the given names. This ignores leaf Properties.
        """
        if not self.has_children():
            raise ValueError("{!r} has no children!".format(self))
        folded_names = [name.casefold() for name in names]
        new_list = []
        merge: dict[str, Property] = {
            name.casefold(): Property(name, [])
            for name in
            names
        }

        for item in self._value[:]:  # type: Property
            if item._folded_name in folded_names:
                merge[item._folded_name]._value.extend(item._value)
            else:
                new_list.append(item)

        for prop_name in names:
            prop = merge[prop_name.casefold()]
            if len(prop._value) > 0:
                new_list.append(prop)

        self._value = new_list

    def ensure_exists(self, key: str) -> 'Property':
        """Ensure a Property group exists with this name, and return it."""
        try:
            return self.find_key(key)
        except NoKeyError:
            prop = Property(key, [])
            self._value.append(prop)
            return prop

    def has_children(self) -> builtins.bool:
        """Does this have child properties?"""
        return type(self._value) is list

    def __repr__(self) -> str:
        return 'Property({0!r}, {1!r})'.format(self.real_name, self._value)

    def __str__(self) -> str:
        return ''.join(self.export())

    def export(self) -> Iterator[str]:
        """Generate the set of strings for a property file.

        Recursively calls itself for all child properties.
        """
        if isinstance(self._value, list):
            if self.name is None:
                # If the name is None, we just output the children
                # without a "Name" { } surround. These Property
                # objects represent the root.
                for prop in self._value:
                    yield from prop.export()
            else:
                assert self.real_name is not None, repr(self)
                yield '"' + self.real_name + '"\n'
                yield '\t{\n'
                yield from (
                    '\t' + line
                    for prop in self._value
                    for line in prop.export()
                )
                yield '\t}\n'
        else:
            # We need to escape quotes and backslashes so they don't get detected.
            assert self.real_name is not None, repr(self)
            yield '"{}" "{}"\n'.format(escape_text(self.real_name), escape_text(self._value))

    def build(self) -> '_Builder':
        """Allows appending a tree to this property in a convenient way.

        Use as follows:
        # doctest: +NORMALIZE_WHITESPACE
        >>> prop = Property('name', [])
        >>> with prop.build() as builder:
        ...     builder.root1('blah')
        ...     builder.root2('blah')
        ...     with builder.subprop:
        ...         subprop = builder.config('value')
        ...         builder['unusual name']('value')
        Property('root1', 'blah')
        Property('root2', 'blah')
        Property('unusual name', 'value')
        >>> print(subprop) # doctest: +NORMALIZE_WHITESPACE
        "config" "value"
        >>> print(prop) # doctest: +NORMALIZE_WHITESPACE
        "name"
            {
            "root1" "blah"
            "root2" "blah"
            "subprop"
                {
                "config" "value"
                "unusual name" "value"
                }
            }

        Return values/results of the context manager are the properties.
        Set names by builder.name, builder['name']. For keywords append '_'.

        Alternatively:
        >>> with Property('name', []).build() as (prop, builder):
        ...     builder.root1('blah')
        ...     builder.root2('blah')
        Property('root1', 'blah')
        Property('root2', 'blah')
        >>> print(repr(prop))
        Property('name', [Property('root1', 'blah'), Property('root2', 'blah')])
        """
        if not self.has_children():
            raise ValueError("{!r} has no children!".format(self))
        return _Builder(self)


class _Builder:
    """Allows constructing property trees using with: chains.

    This is the builder you get directly, which is interacted with for
    all the hierachies.
    """
    def __init__(self, parent: Property):
        self._parents = [parent]

    def __getattr__(self, name: str) -> '_BuilderElem':
        """Accesses by attribute produce a prop with that name.

        For all Python keywords a trailing underscore is ignored.
        """
        return _BuilderElem(self, self._keywords.get(name, name))

    def __getitem__(self, name: str) -> '_BuilderElem':
        """This allows accessing names dynamicallyt easily, or names which aren't identifiers."""
        return _BuilderElem(self, name)

    def __enter__(self) -> '_Builder':
        """Start a property block."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Ends the property block."""
        pass

    def __iter__(self) -> Iterator[Union[Property, '_Builder']]:
        """This allows doing as (x, y) to get the parent."""
        return iter((self._parents[0], self))

    _keywords = {kw + '_': kw for kw in keyword.kwlist}


# noinspection PyProtectedMember
class _BuilderElem:
    """Allows constructing property trees using with: chains.

    This is produced when indexing or accessing attributes on the builder,
    and is then called or used as a context manager to enter the builder.
    """
    def __init__(self, builder: _Builder, name: str):
        self._builder = builder
        self._name = name

    def __call__(self, value: str) -> Property:
        """Add a key-value pair."""
        prop = Property(self._name, value)
        self._builder._parents[-1].append(prop)
        return prop

    def __enter__(self) -> Property:
        """Start a property block."""
        prop = Property(self._name, [])
        self._builder._parents[-1].append(prop)
        self._builder._parents.append(prop)
        return prop

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._builder._parents.pop()
