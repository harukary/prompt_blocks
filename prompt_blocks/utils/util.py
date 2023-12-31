import json, datetime

class CustomEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.indent = 0
    
    def encode(self, obj):
        if isinstance(obj, dict):
            obj_str = "{\n"
            self.indent += 2
            for key, value in obj.items():
                obj_str += " " * self.indent + json.dumps(key) + ": " + self.encode(value) + ",\n"
            obj_str = obj_str[:-2] + "\n"
            self.indent -= 2
            obj_str += " " * self.indent + "}"
            return obj_str
        elif isinstance(obj, list):
            obj_str = "["
            self.indent += 2
            vector = True
            for item in obj:
                if isinstance(item, (list, dict)):
                    vector = False
            if vector:
                if obj:
                    for item in obj:
                        obj_str += self.encode(item) + ","
                    obj_str = obj_str[:-1]
                self.indent -= 2
                obj_str += "]"
            else:
                for item in obj:
                    obj_str += "\n" + " " * self.indent + self.encode(item) + ","
                obj_str = obj_str[:-1] + "\n"
                self.indent -= 2
                obj_str += " " * self.indent + "]"
            return obj_str
        return super().encode(obj)
    
def join_history(messages,mapping=None,dt=False,diff=None):
    # print(messages)
    history_text = ''
    if mapping is None:
        if dt:
            if diff:
                history_text += '\n'.join([
                    f"{h['message']['role']} [{(h['datetime']+datetime.timedelta(**diff)).strftime('%Y/%m/%d %H:%M')}]: {h['message']['content']}"
                    for h in messages if not h['message']['role']=='system']
                )
            else:
                history_text += '\n'.join([
                    f"{h['message']['role']} [{h['datetime'].strftime('%Y/%m/%d %H:%M')}]: {h['message']['content']}"
                    for h in messages if not h['message']['role']=='system']
                )
        else:
            history_text += '\n'.join([
                f"{h['message']['role']}: {h['message']['content']}"
                for h in messages if not h['message']['role']=='system']
            )
    else:
        if dt:
            if diff:
                history_text += '\n'.join([
                    f"{mapping[h['message']['role']]} [{(h['datetime']+datetime.timedelta(**diff)).strftime('%Y/%m/%d %H:%M')}]: {h['message']['content']}"
                    for h in messages if not h['message']['role']=='system']
                )
            else:
                history_text += '\n'.join([
                    f"{mapping[h['message']['role']]} [{h['datetime'].strftime('%Y/%m/%d %H:%M')}]: {h['message']['content']}"
                    for h in messages if not h['message']['role']=='system']
                )
        else:
            history_text += '\n'.join([
                f"{mapping[h['message']['role']]}: {h['message']['content']}"
                for h in messages if not h['message']['role']=='system']
            )
    return history_text