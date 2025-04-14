// Implementation based on gingerBill's
// https://gist.github.com/gingerBill/7282ff54744838c52cc80c559f697051

package handlemap
import "base:builtin"

Handle :: struct {
	generation: u32,
	index:      u32,
}

handle_to_rawptr :: proc(using h: $T/Handle) -> rawptr {
	u: uintptr
	u |= (uintptr(generation) << 32)
	u |= (uintptr(index) << 0)
	return rawptr(u)
}

rawptr_to_handle :: proc(r: rawptr) -> Handle {
	u := uintptr(r)

	return Handle {
		generation = u32(u >> 32),
		index = u32(u & 0xFFFFFFFF)
	}
}

@(private)
Sparse_Index :: struct {
	generation:    u32,
	index_or_next: u32,
}

Handle_Map :: struct($T: typeid) {
	handles:        [dynamic]Handle,
	values:         [dynamic]T,
	sparse_indices: [dynamic]Sparse_Index,
	next:           u32,
}

init :: proc(m: ^$M/Handle_Map($T), allocator := context.allocator) {
	m^ = {}
	m.handles.allocator        = allocator
	m.values.allocator         = allocator
	m.sparse_indices.allocator = allocator
	m.next = 0
}

destroy :: proc(m: ^$M/Handle_Map($T)) {
	m^ = {}
	clear(m)
	delete(m.handles)
	delete(m.values)
	delete(m.sparse_indices)
}

clear :: proc(m: ^$M/Handle_Map($T)) {
	builtin.clear(&m.handles)
	builtin.clear(&m.values)
	builtin.clear(&m.sparse_indices)
	m.next = 0
}

@(require_results)
has_handle :: proc(m: $M/Handle_Map($T), h: $H/Handle) -> bool {
	if h.index < u32(len(m.sparse_indices)) {
		return m.sparse_indices[h.index].generation == h.generation
	}
	return false
}

@(require_results)
get :: proc(m: ^$M/Handle_Map($T), h: $H/Handle) -> (^T, bool) {
	if h.index < u32(len(m.sparse_indices)) {
		entry := m.sparse_indices[h.index]
		if entry.generation == h.generation {
			return &m.values[entry.index_or_next], true
		}
	}
	return nil, false
}

@(require_results)
insert :: proc(m: ^$M/Handle_Map($T), value: T) -> (handle: Handle) {
	if m.next < u32(len(m.sparse_indices)) {
		entry := &m.sparse_indices[m.next]
		assert(entry.generation < max(u32), "Generation sparse indices overflow")

		entry.generation += 1
		handle = Handle{
			generation = entry.generation,
			index = m.next,
		}
		m.next = entry.index_or_next
		entry.index_or_next = u32(len(m.handles))
		append(&m.handles, handle)
		append(&m.values,  value)
	} else {
		assert(m.next < max(u32), "Index sparse indices overflow")

		handle = Handle{
			index = u32(len(m.sparse_indices)),
		}
		append(&m.sparse_indices, Sparse_Index{
			index_or_next = u32(len(m.handles)),
		})
		append(&m.handles, handle)
		append(&m.values,  value)
		m.next += 1
	}
	return
}

remove :: proc(m: ^$M/Handle_Map($T), h: $H/Handle) -> (value: Maybe(T)) {
	if h.index < u32(len(m.sparse_indices)) {
		entry := &m.sparse_indices[h.index]
		if entry.generation != h.generation {
			return
		}
		index := entry.index_or_next
		entry.generation += 1
		entry.index_or_next = m.next
		m.next = h.index
		value = m.values[index]
		unordered_remove(&m.handles, int(index))
		unordered_remove(&m.values,  int(index))
		if index < u32(len(m.handles)) {
			m.sparse_indices[m.handles[index].index].index_or_next = index
		}
	}
	return
}