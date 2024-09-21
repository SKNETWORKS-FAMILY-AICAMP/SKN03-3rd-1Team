document.addEventListener('DOMContentLoaded', function() {
    const table = document.querySelector('table');

    // 테이블 정렬
    table.querySelectorAll('th').forEach(header => {
        header.addEventListener('click', () => {
            const index = Array.from(header.parentElement.children).indexOf(header);
            const rows = Array.from(table.querySelectorAll('tbody tr'));
            rows.sort((rowA, rowB) => {
                const cellA = rowA.children[index].innerText;
                const cellB = rowB.children[index].innerText;
                return cellA.localeCompare(cellB);
            });
            rows.forEach(row => table.querySelector('tbody').appendChild(row));
        });
    });

    // 검색 기능
    const searchInput = document.createElement('input');
    searchInput.setAttribute('type', 'text');
    searchInput.setAttribute('placeholder', 'Search...');
    document.body.insertBefore(searchInput, table);

    searchInput.addEventListener('input', function() {
        const filter = this.value.toLowerCase();
        table.querySelectorAll('tbody tr').forEach(row => {
            const text = row.innerText.toLowerCase();
            row.style.display = text.includes(filter) ? '' : 'none';
        });
    });
});
