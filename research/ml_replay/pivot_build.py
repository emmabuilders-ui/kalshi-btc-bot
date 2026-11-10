# Build the pivot workbook: Data sheet (openpyxl) then pivot tables added via LibreOffice UNO (soffice listening on port 2002), saved as xlsx.
import pandas as pd, uno, os, sys
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from com.sun.star.table import CellAddress, CellRangeAddress
from com.sun.star.beans import PropertyValue
D = pd.read_parquet('work/pivot_long.parquet')
wb = Workbook(); ws = wb.active; ws.title = 'Data'
ws.append(list(D.columns))
for r in D.itertuples(index=False): ws.append(list(r))
for row in ws.iter_rows():
    for c in row: c.font = Font(name='Arial')
for c in ws[1]: c.font = Font(name='Arial', bold=True, color='FFFFFF'); c.fill = PatternFill('solid', fgColor='1F4E78'); c.alignment = Alignment(wrap_text=True)
ws.freeze_panes = 'A2'
for i, w in enumerate([9,15,11,8,12,10,36,10,10,8,10,10,12,10,10,9,12,10,20,30]): ws.column_dimensions[chr(65+i)].width = w
nt = wb.create_sheet('Notes')
for t in open('pivot_notes.txt').read().strip().split('\n'): nt.append([t]); nt.cell(nt.max_row, 1).font = Font(name='Arial')
nt.column_dimensions['A'].width = 160
base = os.path.abspath('work/pivot_base.xlsx'); wb.save(base)
NROW, NCOL = ws.max_row, ws.max_column

ctx = uno.getComponentContext(); res = ctx.ServiceManager.createInstanceWithContext('com.sun.star.bridge.UnoUrlResolver', ctx)
rc = res.resolve('uno:socket,host=127.0.0.1,port=2002;urp;StarOffice.ComponentContext')
desk = rc.ServiceManager.createInstanceWithContext('com.sun.star.frame.Desktop', rc)
pv = lambda n, v: (lambda p: (setattr(p, 'Name', n), setattr(p, 'Value', v), p)[2])(PropertyValue())
doc = desk.loadComponentFromURL(uno.systemPathToFileUrl(base), '_blank', 0, (pv('Hidden', True),))
sheets = doc.Sheets; src = CellRangeAddress(); src.Sheet = 0; src.StartColumn = 0; src.StartRow = 0; src.EndColumn = NCOL - 1; src.EndRow = NROW - 1
E = lambda t, v: uno.Enum(t, v)
def pivot(sheet_name, pos, title, rows=(), cols=(), pages=(), data=(), sort=None, before='Data'):
    sheets.insertNewByName(sheet_name, sheets.getCount())
    sh = sheets.getByName(sheet_name); sh.getCellByPosition(0, 0).setString(title)
    dp = sh.getDataPilotTables(); d = dp.createDataPilotDescriptor(); d.setSourceRange(src)
    F = d.getDataPilotFields()
    def fld(name):
        for i in range(F.getCount()):
            f = F.getByIndex(i)
            if f.getName() == name: return f
        raise KeyError(name)
    for name, sel in pages:
        f = fld(name); f.Orientation = E('com.sun.star.sheet.DataPilotFieldOrientation', 'PAGE')
        if sel: f.SelectedPage = sel
    for name in rows: fld(name).Orientation = E('com.sun.star.sheet.DataPilotFieldOrientation', 'ROW')
    for name in cols: fld(name).Orientation = E('com.sun.star.sheet.DataPilotFieldOrientation', 'COLUMN')
    for name, fn in data:
        f = fld(name); f.Orientation = E('com.sun.star.sheet.DataPilotFieldOrientation', 'DATA'); f.Function = E('com.sun.star.sheet.GeneralFunction', fn)
    a = CellAddress(); a.Sheet = sheets.getCount() and [i for i in range(sheets.getCount()) if sheets.getByIndex(i).getName() == sheet_name][0]; a.Column = 0; a.Row = pos
    dp.insertNewByName(sheet_name.replace(' ', '_'), a, d)
    t = dp.getByName(sheet_name.replace(' ', '_')); df = t.getDataPilotFields()
    for name, sel in pages:
        if not sel: continue
        for i in range(df.getCount()):
            f = df.getByIndex(i)
            if f.getName() == name:
                it = f.getItems()
                for j in range(it.getCount()):
                    x = it.getByIndex(j); x.setPropertyValue('IsHidden', x.getName() != sel)
                f.setPropertyValue('SelectedPage', sel)
    if sort:
        for i in range(df.getCount()):
            f = df.getByIndex(i)
            if f.getName() == sort[0]:
                si = uno.createUnoStruct('com.sun.star.sheet.DataPilotFieldSortInfo'); si.Field = sort[1]; si.IsAscending = False
                si.Mode = uno.getConstantByName('com.sun.star.sheet.DataPilotFieldSortMode.DATA'); f.SortInfo = si; f.HasSortInfo = True
    return sh
pivot('Kalshi weakest', 2, "Kalshi's weakest spots, sorted by Abs % wrong (actual win % minus Kalshi price). Change Month / Period filters to see one month.",
      rows=['Cell (time | price)'], pages=[('Model', 'Kalshi price'), ('Period', None), ('Month', None)],
      data=[('Abs % wrong', 'AVERAGE'), ('% wrong (win - prob, pts)', 'AVERAGE'), ('Actual win %', 'AVERAGE'), ('Kalshi avg price %', 'AVERAGE'), ('Snapshots', 'SUM')],
      sort=('Cell (time | price)', 'Abs % wrong'))
pivot('Kalshi weakest monthly', 2, "Every month x time x price spot, sorted by Kalshi Abs % wrong. Filter Period for Mar-Aug or Sep-Oct.",
      rows=['Cell (month | time | price)'], pages=[('Model', 'Kalshi price'), ('Period', None)],
      data=[('Abs % wrong', 'AVERAGE'), ('% wrong (win - prob, pts)', 'AVERAGE'), ('Actual win %', 'AVERAGE'), ('Kalshi avg price %', 'AVERAGE'), ('Snapshots', 'SUM')],
      sort=('Cell (month | time | price)', 'Abs % wrong'))
pivot('Kalshi grid', 2, "Kalshi % wrong (pts) by price band and minutes left. + = YES underpriced, - = NO underpriced. Change Model to compare any model.",
      rows=['Kalshi price band'], cols=['Minutes left'], pages=[('Model', 'Kalshi price'), ('Period', None), ('Month', None)],
      data=[('% wrong (win - prob, pts)', 'AVERAGE')])
pivot('Models vs Kalshi', 2, "Brier vs Kalshi x1000 by model and month (negative = beats Kalshi). Set Sample to exclude in-sample rows.",
      rows=['Model'], cols=['Month'], pages=[('Time group', None), ('Kalshi price band', None), ('Sample', None)],
      data=[('Brier vs Kalshi x1000', 'AVERAGE')])
pivot('Win pct vs models', 2, "Actual win % vs each model's average probability and its % wrong. Filter by month, time and price.",
      rows=['Model'], pages=[('Month', None), ('Period', None), ('Time group', None), ('Kalshi price band', None)],
      data=[('Actual win %', 'AVERAGE'), ('Avg prob %', 'AVERAGE'), ('% wrong (win - prob, pts)', 'AVERAGE'), ('Abs % wrong', 'AVERAGE'), ('Beats Kalshi (1=yes)', 'SUM'), ('Snapshots', 'SUM')])
out = os.path.abspath('work/kalshi_vs_models_pivot.xlsx')
doc.storeToURL(uno.systemPathToFileUrl(out), (pv('FilterName', 'Calc MS Excel 2007 XML'),)); doc.close(True); print('saved', out)
