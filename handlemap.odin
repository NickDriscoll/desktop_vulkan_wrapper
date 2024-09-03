// Implementation from https://gist.github.com/gingerBill/7282ff54744838c52cc80c559f697051

package desktop_vulkan_wrapper

@(private)
Handle_Map :: struct($T: typeid) {
	handles:        [dynamic]Handle,
	values:         [dynamic]T,
	sparse_indices: [dynamic]Sparse_Index,
	next:           u32,
}

@(private)
Handle :: struct {
	generation: u32,
	index:      u32,
}

@(private)
Sparse_Index :: struct {
	generation:    u32,
	index_or_next: u32,
}

hm_init :: proc(m: ^$M/Handle_Map($T), allocator := context.allocator) {
	m.handles.allocator        = allocator
	m.values.allocator         = allocator
	m.sparse_indices.allocator = allocator
	m.next = 0
}

hm_destroy :: proc(m: ^$M/Handle_Map($T)) {
	clear(m)
	delete(m.handles)
	delete(m.values)
	delete(m.sparse_indices)
}

hm_clear :: proc(m: ^$M/Handle_Map($T)) {
	builtin.clear(&m.handles)
	builtin.clear(&m.values)
	builtin.clear(&m.sparse_indices)
	m.next = 0
}

@(require_results)
hm_has_handle :: proc(m: $M/Handle_Map($T), h: Handle) -> bool {
	if h.index < u32(len(m.sparse_indices)) {
		return m.sparse_indices[h.index].generation == h.generation
	}
	return false
}

@(require_results)
hm_get :: proc(m: ^$M/Handle_Map($T), h: Handle) -> (^T, bool) {
	if h.index < u32(len(m.sparse_indices)) {
		entry := m.sparse_indices[h.index]
		if entry.generation == h.generation {
			return &m.values[entry.index_or_next], true
		}
	}
	return nil, false
}

@(require_results)
hm_insert :: proc(m: ^$M/Handle_Map($T), value: T) -> (handle: Handle) {
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

hm_remove :: proc(m: ^$M/Handle_Map($T), h: Handle) -> (value: Maybe(T)) {
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