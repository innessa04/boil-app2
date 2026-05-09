"""
BOiL – Zagadnienie Pośrednika
Badania Operacyjne i Logistyka

Algorytm:
  1. Macierz zysków jednostkowych Z_ij = C_j - k_ij^t - k_ij^z
  2. Bilansowanie (fikcyjny dostawca FD)
  3. Plan bazowy: metoda wierzchołka NW
  4. Optymalizacja metodą MODI (zmienne dualne α, β + kryterialne δ)
  5. Wizualizacja każdej iteracji w formie tabel
"""

import customtkinter as ctk
import numpy as np
from tkinter import messagebox

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ─── KOLORY ──────────────────────────────────────────────────────────────────
C_BG      = "#0d1117"
C_SIDEBAR = "#161b27"
C_CARD    = "#1a2133"
C_CARD2   = "#1e2640"
C_ACCENT  = "#4f8ef7"
C_GOLD    = "#f5c842"
C_GREEN   = "#22c55e"
C_RED     = "#ef4444"
C_MUTED   = "#8b9ab5"
C_TEXT    = "#e8edf5"
M_BIG     = 99999  # kara za zablokowaną trasę (wstawiana tylko w wierszach D1/D2, nie FD)


# ─── ALGORYTM ────────────────────────────────────────────────────────────────

def nw_corner(supply, demand):
    """Metoda wierzchołka północno-zachodniego."""
    S, C = len(supply), len(demand)
    alloc = np.zeros((S, C))
    s, d = list(supply), list(demand)
    i = j = 0
    while i < S and j < C:
        val = min(s[i], d[j])
        alloc[i, j] = val
        s[i] -= val
        d[j] -= val
        if s[i] == 0:
            i += 1
        if d[j] == 0:
            j += 1
    return alloc


def find_loop(alloc, pivot_i, pivot_j):
    """
    Szuka pętli prostokątnej dla trasy niebazowej (pivot_i, pivot_j).
    Zwraca węzły pętli bez duplikatu węzła startowego.
    """
    S, C = alloc.shape
    basis = {(i, j) for i in range(S) for j in range(C) if alloc[i, j] > 0}
    basis.add((pivot_i, pivot_j))

    def dfs(path, direction):
        if len(path) >= 4 and path[-1] == (pivot_i, pivot_j):
            return path[:-1]
        ci, cj = path[-1]
        if direction == 'row':
            cands = [(ci, jj) for jj in range(C)
                     if (ci, jj) in basis and (ci, jj) not in path[1:]]
        else:
            cands = [(ii, cj) for ii in range(S)
                     if (ii, cj) in basis and (ii, cj) not in path[1:]]
        for nxt in cands:
            result = dfs(path + [nxt], 'col' if direction == 'row' else 'row')
            if result:
                return result
        return None

    return dfs([(pivot_i, pivot_j)], 'row') or dfs([(pivot_i, pivot_j)], 'col')


def modi_optimize(z_matrix, alloc_init):
    """
    Optymalizacja metodą MODI. Zwraca listę iteracji.
    Każda iteracja zawiera: alloc, alpha, beta, delta, pivot, is_optimal.
    """
    S, C = alloc_init.shape
    alloc = alloc_init.copy().astype(float)
    iterations = []

    for it_num in range(100):
        # ── Zmienne dualne α_i, β_j ────────────────────────────────────────
        alpha = [None] * S
        beta  = [None] * C
        alpha[0] = 0

        changed = True
        while changed:
            changed = False
            for i in range(S):
                for j in range(C):
                    if alloc[i, j] > 0:
                        if alpha[i] is not None and beta[j] is None:
                            beta[j] = z_matrix[i, j] - alpha[i]
                            changed = True
                        elif beta[j] is not None and alpha[i] is None:
                            alpha[i] = z_matrix[i, j] - beta[j]
                            changed = True

        alpha = [a if a is not None else 0 for a in alpha]
        beta  = [b if b is not None else 0 for b in beta]

        # ── Zmienne kryterialne δ_ij = Z_ij − α_i − β_j (trasy X) ────────
        delta = np.full((S, C), np.nan)
        for i in range(S):
            for j in range(C):
                if alloc[i, j] == 0:
                    delta[i, j] = z_matrix[i, j] - alpha[i] - beta[j]

        # ── Trasa wejściowa: max δ ─────────────────────────────────────────
        max_delta, pivot_i, pivot_j = -np.inf, -1, -1
        for i in range(S):
            for j in range(C):
                if not np.isnan(delta[i, j]) and delta[i, j] > max_delta:
                    max_delta, pivot_i, pivot_j = delta[i, j], i, j

        is_optimal = (max_delta <= 1e-9)
        iterations.append({
            'iteration':  it_num + 1,
            'alloc':      alloc.copy(),
            'alpha':      list(alpha),
            'beta':       list(beta),
            'delta':      delta.copy(),
            'pivot':      (pivot_i, pivot_j) if not is_optimal else None,
            'is_optimal': is_optimal,
        })

        if is_optimal:
            break

        # ── Pętla prostokątna → korekta planu ─────────────────────────────
        loop = find_loop(alloc, pivot_i, pivot_j)
        if loop is None:
            break

        minus_nodes = [loop[k] for k in range(1, len(loop), 2)]
        theta = min(alloc[ii, jj] for ii, jj in minus_nodes)

        new_alloc = alloc.copy()
        for k, (ii, jj) in enumerate(loop):
            if k % 2 == 0:
                new_alloc[ii, jj] += theta
            else:
                new_alloc[ii, jj] -= theta
        alloc = new_alloc

    return iterations


# ─── INTERFEJS ───────────────────────────────────────────────────────────────

class BOiLApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Zagadnienie Pośrednika — Badania Operacyjne i Logistyka")
        self.geometry("1380x940")
        self.configure(fg_color=C_BG)
        self.entries = {}
        self._build_ui()

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=270, fg_color=C_SIDEBAR, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self._build_sidebar()

        self.scroll = ctk.CTkScrollableFrame(self, fg_color=C_BG)
        self.scroll.grid(row=0, column=1, sticky="nsew")
        self.scroll.grid_columnconfigure(0, weight=1)
        self._build_main()

    # ── SIDEBAR ───────────────────────────────────────────────────────────────

    def _build_sidebar(self):
        def sep():
            ctk.CTkFrame(self.sidebar, height=1, fg_color="#2a3550").pack(
                fill="x", padx=15, pady=10)

        ctk.CTkLabel(self.sidebar, text="BOiL",
                     font=("Courier New", 30, "bold"), text_color=C_ACCENT
                     ).pack(anchor="w", padx=20, pady=(22, 0))
        ctk.CTkLabel(self.sidebar, text="Zagadnienie Pośrednika",
                     font=("Courier New", 10), text_color=C_MUTED
                     ).pack(anchor="w", padx=20, pady=(0, 4))
        sep()

        # Blokowanie trasy odbiorcy
        ctk.CTkLabel(self.sidebar, text="BLOKOWANIE TRASY ODBIORCY",
                     font=("Arial", 10, "bold"), text_color=C_MUTED
                     ).pack(anchor="w", padx=20)
        ctk.CTkLabel(self.sidebar,
                     text="Wybierz odbiorcę, do którego\ntrasa jest zablokowana:",
                     font=("Arial", 10), text_color=C_TEXT, justify="left"
                     ).pack(anchor="w", padx=20, pady=(4, 6))

        self.block_var = ctk.StringVar(value="brak")
        for lbl, val in [("Brak blokady", "brak"), ("Zablokuj → O1", "0"),
                         ("Zablokuj → O2", "1"), ("Zablokuj → O3", "2")]:
            color = C_RED if val != "brak" else C_TEXT
            ctk.CTkRadioButton(self.sidebar, text=lbl, variable=self.block_var,
                               value=val, text_color=color,
                               hover_color="#3a1515" if val != "brak" else C_ACCENT
                               ).pack(anchor="w", padx=20, pady=3)
        sep()

        # Przyciski
        btn = dict(height=42, corner_radius=8, font=("Arial", 12, "bold"))
        ctk.CTkButton(self.sidebar, text="📂  Wczytaj gotowy przykład",
                      command=self._load_example,
                      fg_color="#252f4a", hover_color="#2e3d60",
                      text_color=C_GOLD, **btn).pack(fill="x", padx=15, pady=4)

        ctk.CTkButton(self.sidebar, text="▶  Uruchom algorytm",
                      command=self._run,
                      fg_color=C_ACCENT, hover_color="#3a7be0",
                      text_color="white", **btn).pack(fill="x", padx=15, pady=4)

        ctk.CTkButton(self.sidebar, text="✕  Wyczyść wyniki",
                      command=self._clear_results,
                      fg_color="transparent", hover_color="#1e2230",
                      text_color=C_MUTED, border_width=1, border_color="#2a3550",
                      height=36, corner_radius=8).pack(fill="x", padx=15, pady=4)
        sep()

    # ── OBSZAR GŁÓWNY ─────────────────────────────────────────────────────────

    def _build_main(self):
        card = ctk.CTkFrame(self.scroll, fg_color=C_CARD, corner_radius=12)
        card.pack(fill="x", padx=20, pady=(20, 8))
        ctk.CTkLabel(card, text="Dane wejściowe",
                     font=("Arial", 14, "bold"), text_color=C_TEXT
                     ).pack(anchor="w", padx=20, pady=(14, 2))
        ctk.CTkLabel(card,
                     text="Koszty transportu k_ij  |  Podaż i cena zakupu k^z  |  Popyt i cena sprzedaży C_j",
                     font=("Arial", 10), text_color=C_MUTED
                     ).pack(anchor="w", padx=20, pady=(0, 10))
        self.grid_frame = ctk.CTkFrame(card, fg_color="transparent")
        self.grid_frame.pack(padx=20, pady=(0, 18))
        self._create_input_grid()

        self.results_box = ctk.CTkFrame(self.scroll, fg_color="transparent")
        self.results_box.pack(fill="both", expand=True, padx=20, pady=8)

    def _create_input_grid(self, S=2, C=3):
        for w in self.grid_frame.winfo_children():
            w.destroy()
        self.entries.clear()
        W, FH, FN = 76, ("Arial", 10, "bold"), ("Arial", 10)

        headers = [""] + [f"O{j+1}" for j in range(C)] + ["Podaż", "Cena zakupu k^z"]
        h_bg    = ["transparent"] + [C_ACCENT] * C + [C_GREEN, C_RED]
        for col, (lbl, bg) in enumerate(zip(headers, h_bg)):
            tc = "white" if bg != "transparent" else C_MUTED
            ctk.CTkLabel(self.grid_frame, text=lbl, font=FH, text_color=tc,
                         fg_color=bg, width=W, height=26, corner_radius=5
                         ).grid(row=0, column=col, padx=2, pady=2)

        for i in range(S):
            ctk.CTkLabel(self.grid_frame, text=f"D{i+1}", font=FH,
                         text_color=C_TEXT, width=W
                         ).grid(row=i+1, column=0, padx=2, pady=2)
            for j in range(C):
                e = ctk.CTkEntry(self.grid_frame, width=W, font=FN,
                                 fg_color=C_CARD2, border_color="#3a5090", border_width=1)
                e.grid(row=i+1, column=j+1, padx=2, pady=2)
                self.entries[f"t_{i}_{j}"] = e
            e_s = ctk.CTkEntry(self.grid_frame, width=W, font=FN,
                               fg_color="#122518", border_color=C_GREEN, border_width=1)
            e_s.grid(row=i+1, column=C+1, padx=2, pady=2)
            self.entries[f"s_{i}"] = e_s
            e_b = ctk.CTkEntry(self.grid_frame, width=W, font=FN,
                               fg_color="#251515", border_color=C_RED, border_width=1)
            e_b.grid(row=i+1, column=C+2, padx=2, pady=2)
            self.entries[f"buy_{i}"] = e_b

        ctk.CTkLabel(self.grid_frame, text="Popyt", font=FH,
                     text_color=C_TEXT, width=W
                     ).grid(row=S+1, column=0, padx=2, pady=2)
        for j in range(C):
            e = ctk.CTkEntry(self.grid_frame, width=W, font=FN,
                             fg_color="#122518", border_color=C_GREEN, border_width=1)
            e.grid(row=S+1, column=j+1, padx=2, pady=2)
            self.entries[f"d_{j}"] = e

        ctk.CTkLabel(self.grid_frame, text="Cena sprzed. Cⱼ", font=FH,
                     text_color=C_TEXT, width=W
                     ).grid(row=S+2, column=0, padx=2, pady=2)
        for j in range(C):
            e = ctk.CTkEntry(self.grid_frame, width=W, font=FN,
                             fg_color="#1a2a10", border_color=C_GOLD, border_width=1)
            e.grid(row=S+2, column=j+1, padx=2, pady=2)
            self.entries[f"sell_{j}"] = e

    # ── DANE PRZYKŁADOWE ──────────────────────────────────────────────────────

    def _load_example(self):
        ex = {
            "t_0_0": "8",  "t_0_1": "14", "t_0_2": "17",
            "s_0": "20",   "buy_0": "10",
            "t_1_0": "12", "t_1_1": "9",  "t_1_2": "19",
            "s_1": "30",   "buy_1": "12",
            "d_0": "10",   "d_1": "28",   "d_2": "27",
            "sell_0": "30","sell_1": "25", "sell_2": "30",
        }
        for k, v in ex.items():
            if k in self.entries:
                self.entries[k].delete(0, "end")
                self.entries[k].insert(0, v)

    def _clear_results(self):
        for w in self.results_box.winfo_children():
            w.destroy()

    # ── URUCHOMIENIE ──────────────────────────────────────────────────────────

    def _run(self):
        try:
            S0, C0 = 2, 3
            supply_orig = [float(self.entries[f"s_{i}"].get()) for i in range(S0)]
            demand_orig = [float(self.entries[f"d_{j}"].get()) for j in range(C0)]
            buy  = [float(self.entries[f"buy_{i}"].get())  for i in range(S0)]
            sell = [float(self.entries[f"sell_{j}"].get()) for j in range(C0)]
            trans = [[float(self.entries[f"t_{i}_{j}"].get()) for j in range(C0)]
                     for i in range(S0)]

            # 1. Macierz zysków jednostkowych Z_ij = C_j − k_ij^t − k_ij^z
            z_orig = np.array(
                [[sell[j] - trans[i][j] - buy[i] for j in range(C0)]
                 for i in range(S0)], dtype=float)

            # 2. Blokowanie trasy odbiorcy
            #    Zgodnie z notatkami: -M wstawiamy TYLKO w jednej komórce —
            #    w wierszu FD (fikcyjny dostawca) przy zablokowanym odbiorcy.
            #    D1 i D2 pozostają bez zmian. Dzięki temu FD nie może obsłużyć
            #    zablokowanego odbiorcy, ale D1/D2 mogą — obliczenia są standardowe,
            #    a wynik zależy tylko od tego czy FD miał alokację w tej kolumnie.
            blk = self.block_var.get()
            blocked_col = int(blk) if blk != "brak" else None

            # 3. Bilansowanie — zawsze przez FD
            sum_s = sum(supply_orig)
            sum_d = sum(demand_orig)
            supply = list(supply_orig)
            demand = list(demand_orig)
            row_names = ["D1", "D2"]
            col_names = [f"O{j+1}" for j in range(C0)]

            z_pen = z_orig.copy()

            if sum_d >= sum_s:
                fd_supply = sum_d - sum_s
                supply.append(fd_supply)
                fd_row = np.zeros((1, C0))
                z_pen  = np.vstack([z_pen, fd_row])
                z_orig = np.vstack([z_orig, fd_row])
                row_names.append("FD")
                fd_row_idx = len(supply) - 1
                # -M TYLKO w komórce FD × zablokowany odbiorca
                if blocked_col is not None:
                    z_pen[fd_row_idx, blocked_col] = -M_BIG
            else:
                fo_demand = sum_s - sum_d
                demand.append(fo_demand)
                z_pen  = np.hstack([z_pen,  np.zeros((len(supply), 1))])
                z_orig = np.hstack([z_orig, np.zeros((len(supply), 1))])
                col_names.append("FO")

            # 4. Plan bazowy: metoda wierzchołka NW
            alloc0 = nw_corner(supply, demand)

            # 5. Optymalizacja MODI
            iterations = modi_optimize(z_pen, alloc0)

            # 6. Wyświetlenie
            self._clear_results()
            self._show_z_matrix(z_orig, z_pen, row_names, col_names, blocked_col)
            self._show_iterations(iterations, z_orig, z_pen, row_names, col_names, blocked_col)
            self._show_summary(iterations, z_orig, row_names, col_names,
                               sell, trans, buy, blocked_col)

        except Exception as e:
            import traceback
            messagebox.showerror("Błąd danych",
                                 f"Sprawdź, czy wszystkie pola są wypełnione liczbami.\n\n"
                                 f"{traceback.format_exc()}")

    # ── HELPERS WIZUALIZACJI ──────────────────────────────────────────────────

    def _section(self, text):
        f = ctk.CTkFrame(self.results_box, fg_color="transparent")
        f.pack(fill="x", pady=(16, 3))
        ctk.CTkFrame(f, width=4, height=20, fg_color=C_ACCENT
                     ).pack(side="left", fill="y", padx=(0, 10))
        ctk.CTkLabel(f, text=text, font=("Arial", 13, "bold"), text_color=C_TEXT
                     ).pack(side="left")

    def _card(self):
        c = ctk.CTkFrame(self.results_box, fg_color=C_CARD, corner_radius=10)
        c.pack(fill="x", pady=3)
        return c

    def _draw_table(self, parent, z_show, rows, cols,
                    alloc=None, pivot=None, delta=None,
                    alpha=None, beta=None, blocked_col=None):
        """
        Rysuje tabelę transportową.
        z_show  – macierz zysków do wyświetlenia (z_orig, czytelne wartości)
        blocked_col – indeks zablokowanej kolumny (tylko dla wierszy D*, nie FD)
        """
        W, FH, FN = 78, ("Arial", 9, "bold"), ("Arial", 9)
        n_real = sum(1 for r in rows if r.startswith("D"))  # liczba prawdziwych dostawców
        row_off = 0

        # Wiersz β
        if beta is not None:
            ctk.CTkLabel(parent, text="β →", font=FH, text_color=C_MUTED, width=W
                         ).grid(row=0, column=0, padx=1, pady=1)
            for j, b in enumerate(beta):
                ctk.CTkLabel(parent, text=f"{b:.0f}", font=FH, text_color=C_GOLD,
                             fg_color="#2b2600", width=W, height=24, corner_radius=4
                             ).grid(row=0, column=j+1, padx=1, pady=1)
            row_off = 1

        # Nagłówki kolumn
        ctk.CTkLabel(parent, text="", width=W).grid(row=row_off, column=0, padx=1)
        for j, col in enumerate(cols):
            is_blk = (blocked_col is not None and j == blocked_col)
            bg = "#3a1515" if is_blk else C_CARD2
            tc = C_RED    if is_blk else C_TEXT
            ctk.CTkLabel(parent, text=f"{col} ⛔" if is_blk else col,
                         font=FH, text_color=tc, fg_color=bg,
                         width=W, height=24, corner_radius=4
                         ).grid(row=row_off, column=j+1, padx=1, pady=1)

        for i, row in enumerate(rows):
            is_fd = (row == "FD")

            # Etykieta wiersza
            if alpha is not None:
                ctk.CTkLabel(parent,
                             text=f"{row}\nα={alpha[i]:.0f}",
                             font=FH, text_color=C_GOLD,
                             fg_color="#2b2600", width=W, height=42, corner_radius=4
                             ).grid(row=i+row_off+1, column=0, padx=1, pady=1)
            else:
                ctk.CTkLabel(parent, text=row, font=FH, text_color=C_TEXT,
                             fg_color=C_CARD2, width=W, height=34, corner_radius=4
                             ).grid(row=i+row_off+1, column=0, padx=1, pady=1)

            for j in range(len(cols)):
                # -M tylko w wierszu FD przy zablokowanej kolumnie
                is_blocked = (blocked_col is not None and j == blocked_col and is_fd)
                is_pivot   = (pivot == (i, j))
                has_alloc  = (alloc is not None and alloc[i, j] > 0)

                main = "−M" if is_blocked else (
                    f"Z={z_show[i,j]:.0f}" if z_show is not None else "")

                sub = ""
                if alloc is not None:
                    if alloc[i, j] > 0:
                        sub = f"[{alloc[i,j]:.0f}]"
                    else:
                        sub = "X"
                        if delta is not None and not np.isnan(delta[i, j]):
                            sub += f"  δ={delta[i,j]:.0f}"

                cell_text = (main + "\n" + sub).strip() if sub else main

                if is_blocked:
                    bg, tc = "#2e1111", C_RED
                elif is_pivot:
                    bg, tc = "#122540", C_ACCENT
                elif has_alloc:
                    bg, tc = "#0f2a1a", C_GREEN
                else:
                    bg, tc = "#16202e", C_TEXT

                ctk.CTkLabel(parent, text=cell_text, font=FN, text_color=tc,
                             fg_color=bg, width=W, height=48, corner_radius=4
                             ).grid(row=i+row_off+1, column=j+1, padx=1, pady=1)

    # ── SEKCJA 1 ──────────────────────────────────────────────────────────────

    def _show_z_matrix(self, z_orig, z_pen, row_names, col_names, blocked_col):
        self._section("①  Macierz zysków jednostkowych   Z_ij = C_j − k_ij^t − k_ij^z")
        card = self._card()
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(padx=16, pady=14)
        self._draw_table(inner, z_orig, row_names, col_names, blocked_col=blocked_col)
        if blocked_col is not None:
            ctk.CTkLabel(
                card,
                text=(f"⚠  Trasa zablokowana do O{blocked_col+1}  →  Z[FD, O{blocked_col+1}] = −M  "
                      f"(D1/D2 mogą nadal dostarczać do O{blocked_col+1} bez zmian)"),
                font=("Arial", 10), text_color=C_RED
            ).pack(pady=(0, 10))

    # ── SEKCJA 2 ──────────────────────────────────────────────────────────────

    def _show_iterations(self, iterations, z_orig, z_pen, row_names, col_names, blocked_col):
        self._section("②  Iteracje algorytmu MODI — plan dostaw, zmienne dualne α, β i kryterialne δ")

        for it in iterations:
            num, alloc   = it['iteration'], it['alloc']
            alpha, beta  = it['alpha'],     it['beta']
            delta, pivot = it['delta'],     it['pivot']
            is_opt       = it['is_optimal']

            subtitle = "Plan bazowy (wierzchołek NW)" if num == 1 else "Po zmianie planu (pętla prostokątna)"
            if is_opt:
                status, sc = "✔  PLAN OPTYMALNY — wszystkie δ_ij ≤ 0", C_GREEN
            else:
                pr = row_names[pivot[0]]
                pc = col_names[pivot[1]]
                dv = delta[pivot[0], pivot[1]]
                status = f"↻  Trasa wejściowa: ({pr} → {pc})   δ = {dv:.0f}"
                sc = C_GOLD

            card = ctk.CTkFrame(self.results_box, fg_color=C_CARD, corner_radius=10)
            card.pack(fill="x", pady=4)

            hdr = ctk.CTkFrame(card, fg_color=C_CARD2, corner_radius=8)
            hdr.pack(fill="x", padx=10, pady=(10, 6))
            hr = ctk.CTkFrame(hdr, fg_color="transparent")
            hr.pack(fill="x", padx=12, pady=8)
            ctk.CTkLabel(hr, text=f"Iteracja {num}",
                         font=("Arial", 12, "bold"), text_color=C_ACCENT).pack(side="left")
            ctk.CTkLabel(hr, text=subtitle,
                         font=("Arial", 10), text_color=C_MUTED).pack(side="left", padx=12)
            ctk.CTkLabel(hr, text=status,
                         font=("Arial", 10, "bold"), text_color=sc).pack(side="right")

            tf = ctk.CTkFrame(card, fg_color="transparent")
            tf.pack(padx=14, pady=(4, 4))
            self._draw_table(tf, z_orig, row_names, col_names,
                             alloc=alloc, pivot=pivot, delta=delta,
                             alpha=alpha, beta=beta, blocked_col=blocked_col)

            # Zysk bieżący liczony na z_orig (bez kary M)
            zysk = float(np.sum(alloc * z_orig))
            ctk.CTkLabel(card, text=f"Zysk bieżący planu: {zysk:.0f}",
                         font=("Arial", 10), text_color=C_MUTED
                         ).pack(anchor="e", padx=14, pady=(2, 10))

    # ── SEKCJA 3 ──────────────────────────────────────────────────────────────

    def _show_summary(self, iterations, z_orig, row_names, col_names,
                      sell, trans, buy, blocked_col):
        self._section("③  Plan optymalny — podsumowanie i weryfikacja zysku")
        card = self._card()

        last  = iterations[-1]
        alloc = last['alloc']
        S, C  = alloc.shape
        n_real_rows = sum(1 for r in row_names if r.startswith("D"))
        n_real_cols = min(3, C)   # bez FO jeśli istnieje

        tf = ctk.CTkFrame(card, fg_color="transparent")
        tf.pack(padx=16, pady=14)
        self._draw_table(tf, z_orig, row_names, col_names,
                         alloc=alloc, blocked_col=blocked_col)

        # Zysk końcowy na z_orig
        total_profit = float(np.sum(alloc * z_orig))

        rf = ctk.CTkFrame(card, fg_color="#0c2018", corner_radius=8)
        rf.pack(fill="x", padx=16, pady=(0, 14))
        ctk.CTkLabel(rf, text="ZYSK CAŁKOWITY POŚREDNIKA",
                     font=("Arial", 11), text_color=C_MUTED).pack(pady=(12, 2))
        ctk.CTkLabel(rf, text=f"{total_profit:.0f}",
                     font=("Arial", 30, "bold"), text_color=C_GREEN).pack(pady=(0, 6))

        # Weryfikacja: Ze = Przychód − K^t − K^z
        try:
            rev = sum(alloc[i, j] * sell[j]
                      for i in range(n_real_rows) for j in range(n_real_cols)
                      if alloc[i, j] > 0)
            kt  = sum(alloc[i, j] * trans[i][j]
                      for i in range(n_real_rows) for j in range(n_real_cols)
                      if alloc[i, j] > 0)
            kz  = sum(alloc[i, j] * buy[i]
                      for i in range(n_real_rows) for j in range(n_real_cols)
                      if alloc[i, j] > 0)
            ctk.CTkLabel(
                rf,
                text=(f"Weryfikacja:  Ze = Przychód − Kᵗ − Kᶻ"
                      f" = {rev:.0f} − {kt:.0f} − {kz:.0f} = {rev-kt-kz:.0f}"),
                font=("Courier New", 10), text_color=C_MUTED
            ).pack(pady=(0, 12))
        except Exception:
            pass

        if blocked_col is not None:
            ctk.CTkLabel(
                card,
                text=(f"ℹ  Trasa zablokowana do O{blocked_col+1}: "
                      f"Z[FD, O{blocked_col+1}] = −M, pozostałe komórki kolumny bez zmian."),
                font=("Arial", 10), text_color=C_MUTED
            ).pack(anchor="w", padx=16, pady=(0, 12))


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = BOiLApp()
    app.mainloop()
