use ansi_term::Color;
// use ansi_term::Style;

fn main() {
    println!("");
    println!("");
    println!("{}", Color::Green.blink().paint("Have a nice day!"));
    println!("");
    println!("");
}
